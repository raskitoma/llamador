#!/usr/bin/env bash
# deploy.sh — llamador deployment console
#
# Two modes:
#   • Interactive  — `./deploy.sh`  (no args)  → arrow-key menu
#   • Scripted     — `./deploy.sh <flags>`     → headless deploy (CI, systemd)
#
# Scripted-mode flags:
#   --logs [svc|all]      tail logs after deploying
#   --rebuild             docker build --no-cache --pull
#   --no-model            skip default GGUF download
#   --systemd             install + enable llamador.service
#   --profile <name>      pass through to docker compose (repeatable)
#   --token <hex>         set API_TOKEN
#   --regenerate-token    rotate API_TOKEN
#   -h | --help           print this header

set -euo pipefail

# ----- colors --------------------------------------------------------------
if [[ -t 1 ]]; then
  C0=$'\033[0m'; CD=$'\033[2m'; CB=$'\033[1m'
  CR=$'\033[1;31m'; CG=$'\033[1;32m'; CY=$'\033[1;33m'
  CC=$'\033[1;36m'; CM=$'\033[1;35m'; CK=$'\033[38;5;244m'
else
  C0=; CD=; CB=; CR=; CG=; CY=; CC=; CM=; CK=
fi

say()   { printf "%s[deploy]%s %s\n"  "$CC" "$C0" "$*"; }
ok()    { printf "%s  ok%s %s\n"      "$CG" "$C0" "$*"; }
warn()  { printf "%s  !%s  %s\n"      "$CY" "$C0" "$*"; }
fail()  { printf "%s  x%s  %s\n"      "$CR" "$C0" "$*" >&2; }
hr()    { printf "%s%s%s\n" "$CD" "----------------------------------------------------------------------" "$C0"; }
pause() { echo; printf "%s  press Enter to continue ...%s " "$CK" "$C0"; read -r _; }

cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"
COMPOSE=(docker compose)

# ===========================================================================
#  ARROW-KEY MENU PRIMITIVE — pure bash, no whiptail/dialog dependency.
#  Sets MENU_CHOICE to selected index (0-based) or -1 if user pressed q/ESC.
# ===========================================================================

# usage: menu "Title" "Subtitle (or empty)" option1 option2 ...
menu() {
  local title="$1"; shift
  local subtitle="$1"; shift
  local options=("$@")
  local n=${#options[@]}
  local sel=0
  local key rest

  tput civis 2>/dev/null || true
  trap 'tput cnorm 2>/dev/null || true' RETURN

  while true; do
    clear
    _print_banner "$title"
    if [[ -n "$subtitle" ]]; then
      printf "  %s%s%s\n\n" "$CK" "$subtitle" "$C0"
    fi
    printf "  %sup/down%s navigate   %sEnter%s select   %sq%s quit\n\n" \
      "$CC" "$C0" "$CC" "$C0" "$CC" "$C0"

    for i in "${!options[@]}"; do
      if [[ $i -eq $sel ]]; then
        printf "  %s>>%s %s%s%s\n" "$CC" "$C0" "$CB" "${options[$i]}" "$C0"
      else
        printf "     %s\n" "${options[$i]}"
      fi
    done
    echo

    IFS= read -rsn1 key
    case "$key" in
      $'\x1b')
        IFS= read -rsn2 -t 0.05 rest || rest=""
        case "$rest" in
          '[A'|'OA') ((sel--)); ((sel < 0))   && sel=$((n-1)) ;;
          '[B'|'OB') ((sel++)); ((sel >= n))  && sel=0 ;;
          '[H'|'OH') sel=0 ;;
          '[F'|'OF') sel=$((n-1)) ;;
          '')        MENU_CHOICE=-1; tput cnorm 2>/dev/null || true; return 1 ;;
        esac
        ;;
      ''|$'\n'|$'\r')
        MENU_CHOICE=$sel
        tput cnorm 2>/dev/null || true
        return 0 ;;
      q|Q)
        MENU_CHOICE=-1
        tput cnorm 2>/dev/null || true
        return 1 ;;
      k) ((sel--)); ((sel < 0))  && sel=$((n-1)) ;;
      j) ((sel++)); ((sel >= n)) && sel=0 ;;
      [1-9])
        local k=$((${key} - 1))
        if (( k >= 0 && k < n )); then
          sel=$k; MENU_CHOICE=$sel
          tput cnorm 2>/dev/null || true; return 0
        fi ;;
    esac
  done
}

confirm() {
  local prompt="${1:-Are you sure?}" default="${2:-n}" ans
  local hint="[y/N]"; [[ "$default" == "y" ]] && hint="[Y/n]"
  printf "  %s? %s%s%s %s " "$CY" "$CB" "$prompt" "$C0" "$hint"
  IFS= read -r ans || true
  ans="${ans:-$default}"
  [[ "${ans,,}" =~ ^y(es)?$ ]]
}

_print_banner() {
  local title="$1"
  printf "%s+======================================================================+%s\n" "$CC" "$C0"
  printf "%s|%s  llama  %s%-58s%s%s|%s\n" "$CC" "$C0" "$CB" "$title" "$C0" "$CC" "$C0"
  printf "%s|%s         %sTurboQuant llama.cpp - Qwen3.6 - Pascal-tuned panel%s         %s|%s\n" \
    "$CC" "$C0" "$CK" "$C0" "$CC" "$C0"
  printf "%s+======================================================================+%s\n" "$CC" "$C0"
  _state_line
}

_state_line() {
  local stack="?" gpu="?" temp="?" port="?"
  if docker info >/dev/null 2>&1; then
    local up
    up="$("${COMPOSE[@]}" ps --status running --services 2>/dev/null | wc -l || echo 0)"
    stack="${up} running"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
    temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')C"
  fi
  [[ -f .env ]] && port="$(grep -E '^CADDY_PORT=' .env | cut -d= -f2- || echo 8088)"
  port="${port:-8088}"
  printf "  %sstack:%s %s   %sGPU:%s %s @ %s   %sport:%s %s\n\n" \
    "$CK" "$C0" "$stack" "$CK" "$C0" "$gpu" "$temp" "$CK" "$C0" "$port"
}

# ===========================================================================
#  PHASES — shared by interactive + scripted modes
# ===========================================================================

preflight() {
  command -v docker >/dev/null || { fail "docker not found"; return 2; }
  "${COMPOSE[@]}" version >/dev/null 2>&1 || { fail "docker compose plugin not found"; return 2; }
  docker info >/dev/null 2>&1 || { fail "cannot talk to dockerd (group? permissions?)"; return 2; }
  ok "docker $(docker --version | awk '{print $3}' | tr -d ',')   compose $("${COMPOSE[@]}" version --short 2>/dev/null || echo '?')"

  command -v nvidia-smi >/dev/null || warn "nvidia-smi missing on PATH"
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
    warn "GPU passthrough check failed (NVIDIA Container Toolkit not configured?)"
    cat <<'EOF' >&2
        Install hint:
          sudo apt-get install -y nvidia-container-toolkit
          sudo nvidia-ctk runtime configure --runtime=docker
          sudo systemctl restart docker
EOF
    if ! confirm "Continue without GPU verification" "n"; then return 3; fi
  else
    ok "GPU passthrough OK: $(docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>/dev/null | head -1 | sed 's/^GPU [0-9]*: //; s/ (UUID:.*//')"
  fi
}

scaffold_env() {
  local set_token="${1:-}" regen="${2:-0}"
  if [[ ! -f .env ]]; then
    cp .env.example .env
    ok "created .env from .env.example"
  fi
  if [[ -n "$set_token" ]]; then
    if grep -q '^API_TOKEN=' .env; then
      sed -i "s|^API_TOKEN=.*|API_TOKEN=${set_token}|" .env
    else printf '\nAPI_TOKEN=%s\n' "$set_token" >> .env
    fi
    ok "API_TOKEN set"
  elif [[ "$regen" == "1" ]] || ! grep -qE '^API_TOKEN=.+' .env; then
    local newtok
    newtok="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 48)"
    if grep -q '^API_TOKEN=' .env; then
      sed -i "s|^API_TOKEN=.*|API_TOKEN=${newtok}|" .env
    else printf '\nAPI_TOKEN=%s\n' "$newtok" >> .env
    fi
    ok "generated API_TOKEN (48 chars)"
  fi
  mkdir -p data/models data/config
  chmod 700 data 2>/dev/null || true
}

build_images() {
  local rebuild="${1:-0}"
  local args=()
  [[ "$rebuild" == "1" ]] && args+=(--no-cache --pull)
  export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1
  "${COMPOSE[@]}" build "${args[@]}"
  ok "images built"
}

bring_up() {
  local profiles=("$@")
  local args=()
  for p in "${profiles[@]}"; do args+=(--profile "$p"); done
  "${COMPOSE[@]}" "${args[@]}" up -d
  ok "containers up"
}

wait_health() {
  local svc="$1" tries="${2:-60}" i=0 cid health status
  while (( i < tries )); do
    cid="$("${COMPOSE[@]}" ps -q "$svc" 2>/dev/null || true)"
    [[ -z "$cid" ]] && { sleep 1; ((i++)); continue; }
    status="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || echo unknown)"
    health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo none)"
    if [[ "$health" == "healthy" ]]; then ok "$svc healthy"; return 0; fi
    if [[ "$status" == "running" && "$health" == "none" && $i -gt 5 ]]; then
      ok "$svc running (no healthcheck)"; return 0
    fi
    if [[ "$status" == "exited" || "$status" == "dead" ]]; then
      fail "$svc died -- check: docker compose logs $svc"; return 1
    fi
    printf "."
    sleep 1; ((i++))
  done
  echo; warn "$svc didn't reach healthy in ${tries}s"
  return 0
}

wait_all_health() {
  wait_health backend  60
  wait_health frontend 30
  wait_health caddy    30
  say "waiting for llama-engine (model load takes 1-2 min)"
  wait_health llama-engine 180
}

offer_model() {
  local default
  default="$(grep -E '^MODEL_FILE=' .env | cut -d= -f2-)"
  default="${default:-Qwen3.6-35B-A3B-Q4_K_M.gguf}"
  if [[ -f "data/models/$default" ]]; then
    ok "model present: data/models/$default"
    return 0
  fi
  if confirm "Default model $default not present. Download (~21 GB) now" "y"; then
    ./scripts/pull-model.sh unsloth/Qwen3.6-35B-A3B-GGUF "$default"
  fi
}

print_summary() {
  hr
  local ip port tok
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"; ip="${ip:-localhost}"
  port="$(grep -E '^CADDY_PORT=' .env | cut -d= -f2-)"; port="${port:-8088}"
  tok="$(grep -E '^API_TOKEN=' .env | cut -d= -f2-)"
  printf "  %sControl panel%s  http://%s:%s/\n" "$CB" "$C0" "$ip" "$port"
  printf "  %sOpenAI API%s     http://%s:%s/v1\n" "$CB" "$C0" "$ip" "$port"
  printf "  %sMetrics%s        http://%s:%s/metrics\n" "$CB" "$C0" "$ip" "$port"
  if [[ -n "$tok" ]]; then
    printf "  %sAPI token%s      %s\n" "$CB" "$C0" "$tok"
  fi
}

# ===========================================================================
#  ACTIONS
# ===========================================================================

action_full_deploy() {
  local profiles=("$@")
  hr; say "1/6 preflight";        preflight        || { pause; return 1; }
  hr; say "2/6 env / token";      scaffold_env "" 0
  hr; say "3/6 build images";     build_images 0
  hr; say "4/6 start stack";      bring_up "${profiles[@]}"
  hr; say "5/6 wait for health";  wait_all_health
  hr; say "6/6 model";            offer_model
  print_summary
  pause
}

# helpers used by the toggle menu
_flag()  { local a="$1" needle="$2"; [[ " $a " == *" $needle "* ]] && printf "%s[on]%s" "$CG" "$C0" || printf "%s[off]%s" "$CK" "$C0"; }
_bool()  { [[ "$1" == "1" ]] && printf "%s[on]%s" "$CG" "$C0" || printf "%s[off]%s" "$CK" "$C0"; }
_toggle(){ local a="$1" needle="$2"
  if [[ " $a " == *" $needle "* ]]; then echo "$a" | sed "s/\b$needle\b//; s/  / /g"
  else echo "$a $needle"; fi
}

action_custom_deploy() {
  local profiles_str="" rebuild=0 skip_model=0
  while true; do
    local opts=(
      "Profile: chat (open-webui)        $(_flag "$profiles_str" chat)"
      "Profile: metrics (prom + grafana) $(_flag "$profiles_str" metrics)"
      "Force rebuild (--no-cache --pull)  $(_bool "$rebuild")"
      "Skip model download                $(_bool "$skip_model")"
      "Start deploy with these settings"
      "<- Back"
    )
    if ! menu "Custom deploy" "Toggle items, last item starts the deploy" "${opts[@]}"; then return 0; fi
    case $MENU_CHOICE in
      0) profiles_str="$(_toggle "$profiles_str" chat)" ;;
      1) profiles_str="$(_toggle "$profiles_str" metrics)" ;;
      2) rebuild=$((1 - rebuild)) ;;
      3) skip_model=$((1 - skip_model)) ;;
      4)
        local profs=()
        # shellcheck disable=SC2206
        profs=( $profiles_str )
        hr; preflight        || { pause; return 1; }
        hr; scaffold_env "" 0
        hr; build_images "$rebuild"
        hr; bring_up "${profs[@]}"
        hr; wait_all_health
        if [[ "$skip_model" -eq 0 ]]; then hr; offer_model; fi
        print_summary; pause
        return 0 ;;
      5) return 0 ;;
    esac
  done
}

action_update() {
  if [[ -d .git ]]; then
    say "git pull --ff-only"
    git pull --ff-only || warn "git pull failed (continuing)"
  else
    warn "not a git checkout -- skipping pull"
  fi
  build_images 0
  "${COMPOSE[@]}" up -d --remove-orphans
  wait_all_health
  ok "update complete"; pause
}

action_status() {
  hr; say "containers"
  "${COMPOSE[@]}" ps
  echo
  hr; say "GPU"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv
  else warn "nvidia-smi not found"; fi
  echo
  hr; say "data sizes"
  du -sh data/models data/config 2>/dev/null || true
  pause
}

action_logs_menu() {
  local services=(llama-engine backend frontend caddy "all services" "<- Back")
  if ! menu "Logs" "Pick a service to tail (Ctrl-C to detach)" "${services[@]}"; then return 0; fi
  case $MENU_CHOICE in
    0|1|2|3) clear; "${COMPOSE[@]}" logs -f --tail=200 "${services[$MENU_CHOICE]}" || true ;;
    4)       clear; "${COMPOSE[@]}" logs -f --tail=200 || true ;;
    5)       return 0 ;;
  esac
  pause
}

action_power_menu() {
  local opts=("Stop the stack" "Start the stack" "Restart engine only" "Restart everything" "<- Back")
  if ! menu "Power" "" "${opts[@]}"; then return 0; fi
  case $MENU_CHOICE in
    0) confirm "Stop the entire stack" "n"   && "${COMPOSE[@]}" down ;;
    1) "${COMPOSE[@]}" up -d ;;
    2) "${COMPOSE[@]}" restart llama-engine ;;
    3) confirm "Restart all services" "y" && "${COMPOSE[@]}" restart ;;
  esac
  pause
}

action_models_menu() {
  while true; do
    local opts=(
      "List installed GGUFs"
      "Pull Qwen3.6-35B-A3B Q4_K_M (default, ~21 GB)"
      "Pull Qwen3.6-35B-A3B UD-Q4_K_XL (~22 GB)"
      "Pull custom (you'll be prompted)"
      "Delete a model"
      "<- Back"
    )
    if ! menu "Models" "Files live in ./data/models" "${opts[@]}"; then return 0; fi
    case $MENU_CHOICE in
      0) hr; ls -lh data/models/*.gguf 2>/dev/null || warn "no GGUFs yet"; pause ;;
      1) hr; ./scripts/pull-model.sh unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf; pause ;;
      2) hr; ./scripts/pull-model.sh unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; pause ;;
      3)
         hr
         printf "  HF repo (e.g. unsloth/Qwen3.6-35B-A3B-GGUF): "; read -r repo
         printf "  filename                                   : "; read -r fn
         ./scripts/pull-model.sh "$repo" "$fn"; pause ;;
      4)
         hr
         mapfile -t files < <(ls data/models/*.gguf 2>/dev/null | xargs -n1 basename)
         if [[ ${#files[@]} -eq 0 ]]; then warn "no GGUFs"; pause; continue; fi
         files+=("<- Cancel")
         menu "Delete a model" "irreversible" "${files[@]}" || continue
         [[ $MENU_CHOICE -eq $((${#files[@]}-1)) ]] && continue
         local pick="${files[$MENU_CHOICE]}"
         confirm "Delete data/models/$pick" "n" && rm -f "data/models/$pick" && ok "deleted $pick"
         pause ;;
      5) return 0 ;;
    esac
  done
}

action_token_menu() {
  local opts=("Show current token" "Generate a new token" "<- Back")
  if ! menu "API token" "" "${opts[@]}"; then return 0; fi
  case $MENU_CHOICE in
    0) hr; grep -E '^API_TOKEN=' .env || warn "no token set"; pause ;;
    1) hr; scaffold_env "" 1; grep -E '^API_TOKEN=' .env; warn "Existing API clients will need the new value."; pause ;;
  esac
}

action_systemd_menu() {
  local unit=/etc/systemd/system/llamador.service
  local installed="not installed"
  [[ -f "$unit" ]] && installed="installed"
  local opts=("Install + enable (state: $installed)" "Disable + remove" "Show status" "<- Back")
  if ! menu "systemd unit" "Survives host reboots" "${opts[@]}"; then return 0; fi
  case $MENU_CHOICE in
    0) hr; install_systemd; pause ;;
    1) hr; remove_systemd;  pause ;;
    2) hr; sudo systemctl status llamador --no-pager 2>&1 | head -25 || warn "not installed"; pause ;;
  esac
}

install_systemd() {
  command -v systemctl >/dev/null || { warn "no systemctl on this host"; return 1; }
  [[ -f "$ROOT/llamador.service" ]] || { fail "missing $ROOT/llamador.service"; return 1; }
  sudo install -m 0644 "$ROOT/llamador.service" /etc/systemd/system/llamador.service
  sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=${ROOT}|" /etc/systemd/system/llamador.service
  sudo sed -i "s|^User=.*|User=${USER}|"                        /etc/systemd/system/llamador.service
  sudo sed -i "s|^Group=.*|Group=$(id -gn)|"                    /etc/systemd/system/llamador.service
  sudo systemctl daemon-reload
  sudo systemctl enable llamador.service
  ok "installed and enabled. Start with: sudo systemctl start llamador"
}

remove_systemd() {
  command -v systemctl >/dev/null || { warn "no systemctl"; return 1; }
  if ! confirm "Disable + remove the systemd unit" "n"; then return 0; fi
  sudo systemctl disable --now llamador.service 2>/dev/null || true
  sudo rm -f /etc/systemd/system/llamador.service
  sudo systemctl daemon-reload
  ok "removed"
}

# ===========================================================================
#  MAIN MENU LOOP
# ===========================================================================

main_menu() {
  while true; do
    local opts=(
      "Full deploy   - preflight, build, up, wait, model"
      "Custom deploy - toggle profiles, rebuild, skip model"
      "Update        - git pull, rebuild, restart"
      "Status        - containers, GPU, disk"
      "Logs ...      - tail any service"
      "Power ...     - start / stop / restart"
      "Models ...    - list / pull / delete GGUFs"
      "API token ... - show or rotate"
      "systemd ...   - install / remove unit"
      "Quit"
    )
    if ! menu "llamador deploy" "" "${opts[@]}"; then clear; exit 0; fi
    case $MENU_CHOICE in
      0) action_full_deploy ;;
      1) action_custom_deploy ;;
      2) action_update ;;
      3) action_status ;;
      4) action_logs_menu ;;
      5) action_power_menu ;;
      6) action_models_menu ;;
      7) action_token_menu ;;
      8) action_systemd_menu ;;
      9) clear; exit 0 ;;
    esac
  done
}

# ===========================================================================
#  SCRIPTED MODE (preserves headless interface for CI/systemd)
# ===========================================================================

scripted_mode() {
  local DO_LOGS="" LOGS_TARGET="llama-engine" DO_REBUILD=0 DO_MODEL=1
  local DO_SYSTEMD=0 DO_REGEN=0 SET_TOKEN="" PROFILES=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --logs)
        DO_LOGS=1
        if [[ ${2:-} && ${2:0:2} != "--" ]]; then LOGS_TARGET="$2"; shift; fi ;;
      --rebuild)            DO_REBUILD=1 ;;
      --no-model|--skip-model) DO_MODEL=0 ;;
      --systemd)            DO_SYSTEMD=1 ;;
      --regenerate-token)   DO_REGEN=1 ;;
      --token)              SET_TOKEN="${2:-}"; shift ;;
      --profile)            PROFILES+=("${2:-}"); shift ;;
      -h|--help)            sed -n '2,17p' "$0"; exit 0 ;;
      *) fail "unknown arg: $1"; exit 1 ;;
    esac
    shift
  done

  hr; say "1/6 preflight";        preflight                || exit 2
  hr; say "2/6 env / token";      scaffold_env "$SET_TOKEN" "$DO_REGEN"
  hr; say "3/6 build images";     build_images "$DO_REBUILD"
  hr; say "4/6 start stack";      bring_up "${PROFILES[@]}"
  hr; say "5/6 wait for health";  wait_all_health
  if [[ "$DO_MODEL" -eq 1 ]]; then hr; say "6/6 model"; offer_model
  else                              warn "skipping model download (--no-model)"
  fi
  [[ "$DO_SYSTEMD" -eq 1 ]] && { hr; install_systemd; }
  print_summary
  if [[ -n "$DO_LOGS" ]]; then
    hr; say "tailing logs: ${LOGS_TARGET}    (ctrl-c to detach; stack keeps running)"
    if [[ "$LOGS_TARGET" == "all" ]]; then
      exec "${COMPOSE[@]}" logs -f --tail=100
    else
      exec "${COMPOSE[@]}" logs -f --tail=100 "$LOGS_TARGET"
    fi
  fi
}

# ===========================================================================
#  ENTRY
# ===========================================================================

if [[ $# -gt 0 ]]; then
  scripted_mode "$@"
else
  if [[ ! -t 0 || ! -t 1 ]]; then
    fail "interactive menu needs a TTY. Pass flags to use scripted mode (try --help)."
    exit 1
  fi
  main_menu
fi

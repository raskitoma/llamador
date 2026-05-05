# Security model

`llamador` is built for **LAN-trust** out of the box: the people who can reach Caddy on `:8088` can change the model, restart the engine, and trigger image rebuilds. That's intentional for a homelab tool.

If you want to expose it more broadly, layer the following on top.

## 1. Bearer token on the control API

```env
# .env
API_TOKEN=<a long random string>
```

The backend will then require `Authorization: Bearer <token>` on every `/api/*` endpoint. The frontend doesn't currently send a token automatically — if you set this, supply the header from your own client (curl, your own UI, etc.). Adapting the SPA to read the token from `localStorage` is two lines if you want it.

## 2. Reverse proxy auth

Edit `caddy/Caddyfile` to add HTTP basic auth or forward auth in front of `/api`. The simplest:

```caddyfile
handle_path /api/* {
    basic_auth {
        admin $2a$14$<bcrypt-hash>
    }
    reverse_proxy backend:8000
}
```

Generate the bcrypt hash with `caddy hash-password`.

## 3. The Docker socket bomb

The backend mounts `/var/run/docker.sock` so it can `restart`, `build`, etc. **That's effectively root on the host.** Treat the backend container as privileged:

- Don't expose `/api` to the public internet without auth
- Don't run untrusted models (a malicious GGUF can't escape llama.cpp easily, but caution is free)
- Consider running the stack inside a VM if your threat model includes anyone other than you

If you want to drop the socket access, you can — you'll lose:
- _Restart engine_ button
- _Rebuild image_ button
- Live engine log streaming

…but the model picker, config editor, and HF model download still work.

## 4. Production-style TLS

Caddy auto-provisions Let's Encrypt certs if you give it a real domain. Edit `caddy/Caddyfile`:

```caddyfile
nevermind.example.com {
    handle_path /api/* { reverse_proxy backend:8000 }
    handle_path /v1/*  { reverse_proxy llama-engine:8080 }
    handle             { reverse_proxy frontend:80 }
}
```

…and remove the `auto_https off` line in the global block. Open ports 80 and 443 to Caddy.

## 5. Don't commit secrets

The repo's `.gitignore` excludes `.env`. Double-check before pushing:

```bash
git ls-files | grep -E '\.env(?!\.example)'
# nothing should match
```

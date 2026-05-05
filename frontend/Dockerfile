# syntax=docker/dockerfile:1.7
# A no-build static frontend served by nginx-alpine.
# Tailwind + Alpine come from CDN, so there's nothing to compile here.

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css /usr/share/nginx/html/

HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

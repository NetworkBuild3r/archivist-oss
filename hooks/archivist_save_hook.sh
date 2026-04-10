#!/usr/bin/env bash
# archivist_save_hook.sh — Auto-save the current session context to Archivist.
#
# Calls archivist_session_end to summarize and persist the active session
# as a durable memory. Safe to call multiple times — the server aggregates
# trajectories and only stores meaningful deltas.
#
# Environment variables:
#   ARCHIVIST_URL     — MCP endpoint (default: http://localhost:3100)
#   AGENT_ID          — Agent identifier (default: claude)
#   SESSION_ID        — Session identifier (default: auto-<timestamp>)
#   ARCHIVIST_API_KEY — Optional API key for auth
#
# Usage in Claude Code (.claude/settings.json):
#   "hooks": {
#     "Stop": [{ "command": "bash hooks/archivist_save_hook.sh" }]
#   }
#
# Usage as periodic cron (every 15 min):
#   */15 * * * * cd /path/to/project && bash hooks/archivist_save_hook.sh

set -euo pipefail

ARCHIVIST_URL="${ARCHIVIST_URL:-http://localhost:3100}"
AGENT_ID="${AGENT_ID:-claude}"
SESSION_ID="${SESSION_ID:-auto-$(date +%Y%m%d-%H%M%S)}"

_json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")'
}

PAYLOAD=$(printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"archivist_session_end","arguments":{"agent_id":%s,"session_id":%s,"store_as_memory":true}}}' \
  "$(_json_escape "$AGENT_ID")" \
  "$(_json_escape "$SESSION_ID")")

CURL_ARGS=(
  -s -w '\n%{http_code}'
  -X POST "${ARCHIVIST_URL}/mcp"
  -H 'Content-Type: application/json'
  -d "$PAYLOAD"
)

if [ -n "${ARCHIVIST_API_KEY:-}" ]; then
  CURL_ARGS+=(-H "Authorization: Bearer ${ARCHIVIST_API_KEY}")
fi

RESPONSE=$(curl "${CURL_ARGS[@]}" 2>/dev/null) || {
  echo "[archivist_save_hook] WARNING: Archivist not reachable at $ARCHIVIST_URL" >&2
  exit 0
}

HTTP_CODE=$(printf '%s' "$RESPONSE" | tail -1)
BODY=$(printf '%s' "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
  echo "[archivist_save_hook] Session saved: agent=$AGENT_ID session=$SESSION_ID"
else
  echo "[archivist_save_hook] WARNING: HTTP $HTTP_CODE from Archivist" >&2
  echo "$BODY" >&2
fi

#!/usr/bin/env bash
# archivist_precompact_hook.sh — Emergency save before context compaction.
#
# Calls archivist_session_end (to persist trajectories) and then
# archivist_compress (to create a structured summary of recent memories)
# before the context window shrinks. This prevents information loss
# during compaction events.
#
# Environment variables:
#   ARCHIVIST_URL     — MCP endpoint (default: http://localhost:3100)
#   AGENT_ID          — Agent identifier (default: claude)
#   SESSION_ID        — Session identifier (default: auto-<timestamp>)
#   NAMESPACE         — Memory namespace (default: "")
#   ARCHIVIST_API_KEY — Optional API key for auth
#
# Usage in Claude Code (.claude/settings.json):
#   "hooks": {
#     "Compact": [{ "command": "bash hooks/archivist_precompact_hook.sh" }]
#   }

set -euo pipefail

ARCHIVIST_URL="${ARCHIVIST_URL:-http://localhost:3100}"
AGENT_ID="${AGENT_ID:-claude}"
SESSION_ID="${SESSION_ID:-auto-$(date +%Y%m%d-%H%M%S)}"
NAMESPACE="${NAMESPACE:-}"

_json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")'
}

_mcp_call() {
  local payload="$1"

  local -a curl_args=(
    -s -X POST "${ARCHIVIST_URL}/mcp"
    -H 'Content-Type: application/json'
    -d "$payload"
  )
  if [ -n "${ARCHIVIST_API_KEY:-}" ]; then
    curl_args+=(-H "Authorization: Bearer ${ARCHIVIST_API_KEY}")
  fi

  curl "${curl_args[@]}" 2>/dev/null
}

# Step 1: Save the session
echo "[archivist_precompact] Step 1/2: Saving session..."

PAYLOAD=$(printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"archivist_session_end","arguments":{"agent_id":%s,"session_id":%s,"store_as_memory":true}}}' \
  "$(_json_escape "$AGENT_ID")" \
  "$(_json_escape "$SESSION_ID")")

_mcp_call "$PAYLOAD" >/dev/null || {
  echo "[archivist_precompact] WARNING: session_end failed, continuing..." >&2
}

# Step 2: Retrieve recent memory IDs and compress them
echo "[archivist_precompact] Step 2/2: Compressing recent memories..."

RECALL_PAYLOAD=$(printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"archivist_recall","arguments":{"agent_id":%s,"namespace":%s,"query":"recent session context and decisions","limit":20}}}' \
  "$(_json_escape "$AGENT_ID")" \
  "$(_json_escape "$NAMESPACE")")

RECALL_RESULT=$(_mcp_call "$RECALL_PAYLOAD") || {
  echo "[archivist_precompact] WARNING: recall failed, skipping compress" >&2
  exit 0
}

MEMORY_IDS=$(printf '%s' "$RECALL_RESULT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    contents = data.get('result', {}).get('content', [])
    for c in contents:
        text = c.get('text', '')
        try:
            parsed = json.loads(text)
            sources = parsed.get('sources', [])
            ids = [s['id'] for s in sources if 'id' in s]
            print(json.dumps(ids))
            sys.exit(0)
        except (json.JSONDecodeError, KeyError):
            pass
    print('[]')
except Exception:
    print('[]')
" 2>/dev/null)

if [ "$MEMORY_IDS" = "[]" ] || [ -z "$MEMORY_IDS" ]; then
  echo "[archivist_precompact] No recent memories to compress."
  exit 0
fi

COMPRESS_PAYLOAD=$(printf '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"archivist_compress","arguments":{"agent_id":%s,"namespace":%s,"memory_ids":%s,"format":"structured"}}}' \
  "$(_json_escape "$AGENT_ID")" \
  "$(_json_escape "$NAMESPACE")" \
  "$MEMORY_IDS")

_mcp_call "$COMPRESS_PAYLOAD" >/dev/null || {
  echo "[archivist_precompact] WARNING: compress failed" >&2
  exit 0
}

echo "[archivist_precompact] Pre-compaction save complete: agent=$AGENT_ID session=$SESSION_ID"

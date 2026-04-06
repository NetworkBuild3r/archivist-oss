#!/bin/bash
# Tailscale helper for Archivist
# Manages the Tailscale container and helps configure vLLM endpoint

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_status() {
    echo -e "${GREEN}=== Tailscale Status ===${NC}"
    if docker compose ps tailscale | grep -q "Up"; then
        echo -e "${GREEN}✓ Tailscale container is running${NC}"
        echo ""
        docker exec archivist-tailscale-1 tailscale status 2>/dev/null || echo "Waiting for Tailscale to initialize..."
    else
        echo -e "${RED}✗ Tailscale container is not running${NC}"
        echo "Run: docker compose up -d tailscale"
    fi
}

find_vllm() {
    echo -e "${GREEN}=== Finding vLLM Server ===${NC}"
    echo "Searching for vLLM endpoints in your Tailscale network..."
    echo ""
    
    # Get all tailscale IPs/hostnames
    docker exec archivist-tailscale-1 tailscale status --json 2>/dev/null | \
        jq -r '.Peer[] | select(.HostName != null) | "\(.HostName) (\(.TailscaleIPs[0]))"' | \
        while read -r line; do
            echo "  Found: $line"
        done
    
    echo ""
    echo -e "${YELLOW}To test LLM proxy (LiteLLM / vLLM) from archivist:${NC}"
    echo "  $0 test <hostname> [port]     # default port 8000; use 80 for home k3s LiteLLM"
    echo ""
    echo -e "${YELLOW}Update .env and compose:${NC}"
    echo "  LLM_URL=http://<hostname>[:port]   # no /v1 suffix"
    echo "  OPENCLAW_VLLM_TSIP=<100.x>        # if using extra_hosts in docker-compose.yml"
}

test_vllm() {
    if [ -z "$1" ]; then
        echo -e "${RED}Usage: $0 test <hostname>${NC}"
        exit 1
    fi
    
    HOSTNAME=$1
    PORT=${2:-8000}
    
    echo -e "${GREEN}=== Testing vLLM at $HOSTNAME:$PORT ===${NC}"
    echo "Testing from archivist container via Tailscale..."
    docker exec archivist-archivist-1 python3 -c "
import httpx
try:
    with httpx.Client(timeout=10) as client:
        print('Testing /v1/models...')
        resp = client.get('http://$HOSTNAME:$PORT/v1/models')
        if resp.status_code == 200:
            models = resp.json().get('data', [])
            print(f'✓ Success! Found {len(models)} models:')
            for m in models[:5]:
                print(f'  - {m.get(\"id\", \"unknown\")}')
        else:
            print(f'✗ HTTP {resp.status_code}')
            print(resp.text[:200])
except Exception as e:
    print(f'✗ Error: {e}')
" 2>&1
}

update_env() {
    if [ -z "$1" ]; then
        echo -e "${RED}Usage: $0 update <hostname>${NC}"
        exit 1
    fi
    
    HOSTNAME=$1
    PORT=${2:-8000}
    
    echo -e "${GREEN}=== Updating .env ===${NC}"
    sed -i "s|^LLM_URL=.*|LLM_URL=http://$HOSTNAME:$PORT|" .env
    echo -e "${GREEN}✓ Updated LLM_URL=http://$HOSTNAME:$PORT${NC}"
    echo ""
    echo "Restart archivist to apply:"
    echo "  docker compose restart archivist"
}

case "${1:-status}" in
    status)
        show_status
        ;;
    find)
        find_vllm
        ;;
    test)
        test_vllm "$2" "$3"
        ;;
    update)
        update_env "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {status|find|test <hostname>|update <hostname>}"
        echo ""
        echo "Commands:"
        echo "  status  - Show Tailscale connection status"
        echo "  find    - List all devices on your Tailscale network"
        echo "  test    - Test vLLM connectivity: $0 test <hostname> [port]"
        echo "  update  - Update .env with vLLM hostname: $0 update <hostname> [port]"
        exit 1
        ;;
esac

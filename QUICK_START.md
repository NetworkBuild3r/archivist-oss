# Quick Start: Benchmark with qwen3 over Tailscale

## Everything is Ready! 🚀

Your Archivist benchmark is fully configured to use `qwen3` via Tailscale instead of xAI.

### Current Status

- ✅ Tailscale container running and connected
- ✅ Archivist container using Tailscale network
- ✅ Benchmark configured for qwen3 model
- ⏳ Waiting for vLLM service to come online (ArgoCD sync)

### Monitor vLLM Status

```bash
cd /opt/appdata/archivist-oss

# Auto-monitor and test when available
./monitor-vllm.sh
```

This will check every 10 seconds and automatically test connectivity when vLLM comes online.

### Manual Testing

```bash
# Check Tailscale network
docker exec archivist-tailscale-1 tailscale status

# Test vLLM connectivity
./tailscale-helper.sh test openclaw-vllm-corp

# Or test the other server
./tailscale-helper.sh test vllm-mm-dev
```

### Run Benchmarks

Once vLLM is accessible:

```bash
# Full pipeline evaluation (no LLM refinement for speed)
python -m benchmarks.pipeline.evaluate --no-refine --output .benchmarks/pipeline.json

# Quick test (limited questions)
python -m benchmarks.pipeline.evaluate --variant vector_only --no-refine --limit 10

# Single variant
python -m benchmarks.pipeline.evaluate --variant full_pipeline --no-refine

# Scale sweep across corpus sizes
python -m benchmarks.pipeline.evaluate --scale-sweep --variants vector_only,full_pipeline --no-refine
```

### Configuration Files

- **`.env`** - LLM URL, model, and Tailscale auth key
- **`docker-compose.yml`** - Tailscale + Archivist network config
- **`tailscale-helper.sh`** - Test and manage connections
- **`monitor-vllm.sh`** - Auto-monitor vLLM availability

### Documentation

- **`VLLM_STATUS.md`** - Current status and root cause resolution
- **`TAILSCALE_STATUS.md`** - Complete setup status
- **`docs/TAILSCALE_SETUP.md`** - Full setup guide
- **`docs/BENCHMARKS.md`** - Benchmark documentation

### Network Details

**Tailscale Network**: `tail016335.ts.net`

**Available vLLM Servers**:
- `openclaw-vllm-corp.tail016335.ts.net` (100.93.213.109)
- `vllm-mm-dev.tail016335.ts.net` (100.84.142.112)

**Currently Configured** (in `.env`):
```bash
LLM_URL=http://openclaw-vllm-corp.tail016335.ts.net:8000
LLM_MODEL=qwen3
```

### Troubleshooting

If vLLM doesn't connect after ArgoCD sync:

```bash
# Check vLLM device status
docker exec archivist-tailscale-1 tailscale status | grep vllm

# View archivist logs
docker logs archivist-archivist-1 | tail -50

# Restart archivist if needed
docker compose restart archivist
```

### Switch to Different vLLM Server

```bash
# Update to use vllm-mm-dev instead
./tailscale-helper.sh update vllm-mm-dev

# Restart archivist
docker compose restart archivist
```

---

**What Changed**: Switched from xAI Grok to your Tailscale-connected vLLM running qwen3!

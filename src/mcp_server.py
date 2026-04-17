# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('mcp_server.foo') patches
# archivist.app.mcp_server.foo (same module object).
import sys
import archivist.app.mcp_server as _real
sys.modules[__name__] = _real

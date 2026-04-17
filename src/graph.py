# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('graph.foo') patches
# archivist.storage.graph.foo (same module object).
import sys
import archivist.storage.graph as _real
sys.modules[__name__] = _real

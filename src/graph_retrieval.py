# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('graph_retrieval.foo') patches
# archivist.storage.graph_retrieval.foo (same module object).
import sys
import archivist.storage.graph_retrieval as _real
sys.modules[__name__] = _real

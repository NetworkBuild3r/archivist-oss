# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('hot_cache.foo') patches
# archivist.retrieval.hot_cache.foo (same module object).
import sys
import archivist.retrieval.hot_cache as _real
sys.modules[__name__] = _real

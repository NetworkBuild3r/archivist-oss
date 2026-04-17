# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('fts_search.foo') patches
# archivist.storage.fts_search.foo (same module object).
import sys
import archivist.storage.fts_search as _real
sys.modules[__name__] = _real

# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('query_intent.foo') patches
# archivist.retrieval.query_intent.foo (same module object).
import sys
import archivist.retrieval.query_intent as _real
sys.modules[__name__] = _real

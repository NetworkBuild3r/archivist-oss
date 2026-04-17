# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('rlm_retriever.foo') patches
# archivist.retrieval.rlm_retriever.foo (same module object).
import sys
import archivist.retrieval.rlm_retriever as _real
sys.modules[__name__] = _real

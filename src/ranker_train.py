# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('ranker_train.foo') patches
# archivist.retrieval.ranker_train.foo (same module object).
import sys
import archivist.retrieval.ranker_train as _real
sys.modules[__name__] = _real

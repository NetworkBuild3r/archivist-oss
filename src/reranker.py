# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('reranker.foo') patches
# archivist.retrieval.reranker.foo (same module object).
import sys

import archivist.retrieval.reranker as _real

sys.modules[__name__] = _real

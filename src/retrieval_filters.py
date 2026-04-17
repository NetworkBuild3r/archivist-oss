# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('retrieval_filters.foo') patches
# archivist.retrieval.retrieval_filters.foo (same module object).
import sys

import archivist.retrieval.retrieval_filters as _real

sys.modules[__name__] = _real

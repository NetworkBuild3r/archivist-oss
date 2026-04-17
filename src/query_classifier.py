# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('query_classifier.foo') patches
# archivist.retrieval.query_classifier.foo (same module object).
import sys

import archivist.retrieval.query_classifier as _real

sys.modules[__name__] = _real

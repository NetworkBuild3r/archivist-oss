# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('query_expansion.foo') patches
# archivist.retrieval.query_expansion.foo (same module object).
import sys

import archivist.retrieval.query_expansion as _real

sys.modules[__name__] = _real

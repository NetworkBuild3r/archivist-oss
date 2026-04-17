# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('ranker.foo') patches
# archivist.retrieval.ranker.foo (same module object).
import sys

import archivist.retrieval.ranker as _real

sys.modules[__name__] = _real

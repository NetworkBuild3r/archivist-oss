# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('retrieval_log.foo') patches
# archivist.retrieval.retrieval_log.foo (same module object).
import sys

import archivist.retrieval.retrieval_log as _real

sys.modules[__name__] = _real

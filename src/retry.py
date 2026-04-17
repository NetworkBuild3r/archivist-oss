# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('retry.foo') patches
# archivist.utils.retry.foo (same module object).
import sys

import archivist.utils.retry as _real

sys.modules[__name__] = _real

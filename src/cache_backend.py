# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('cache_backend.foo') patches
# archivist.storage.cache_backend.foo (same module object).
import sys

import archivist.storage.cache_backend as _real

sys.modules[__name__] = _real

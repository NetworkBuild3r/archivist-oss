# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('qdrant.foo') patches
# archivist.storage.qdrant.foo (same module object).
import sys

import archivist.storage.qdrant as _real

sys.modules[__name__] = _real

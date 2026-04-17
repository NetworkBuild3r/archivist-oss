# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('collection_router.foo') patches
# archivist.storage.collection_router.foo (same module object).
import sys

import archivist.storage.collection_router as _real

sys.modules[__name__] = _real

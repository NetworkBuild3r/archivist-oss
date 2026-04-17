# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('compressed_index.foo') patches
# archivist.storage.compressed_index.foo (same module object).
import sys

import archivist.storage.compressed_index as _real

sys.modules[__name__] = _real

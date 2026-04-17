# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('tiering.foo') patches
# archivist.write.tiering.foo (same module object).
import sys

import archivist.write.tiering as _real

sys.modules[__name__] = _real

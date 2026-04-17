# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('memory_lifecycle.foo') patches
# archivist.lifecycle.memory_lifecycle.foo (same module object).
import sys

import archivist.lifecycle.memory_lifecycle as _real

sys.modules[__name__] = _real

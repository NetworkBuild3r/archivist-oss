# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('curator_queue.foo') patches
# archivist.lifecycle.curator_queue.foo (same module object).
import sys

import archivist.lifecycle.curator_queue as _real

sys.modules[__name__] = _real

# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('merge.foo') patches
# archivist.lifecycle.merge.foo (same module object).
import sys

import archivist.lifecycle.merge as _real

sys.modules[__name__] = _real

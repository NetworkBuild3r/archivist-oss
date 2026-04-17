# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('dashboard.foo') patches
# archivist.app.dashboard.foo (same module object).
import sys

import archivist.app.dashboard as _real

sys.modules[__name__] = _real

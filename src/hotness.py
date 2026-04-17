# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('hotness.foo') patches
# archivist.core.hotness.foo (same module object).
import sys

import archivist.core.hotness as _real

sys.modules[__name__] = _real

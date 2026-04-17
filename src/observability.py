# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('observability.foo') patches
# archivist.core.observability.foo (same module object).
import sys

import archivist.core.observability as _real

sys.modules[__name__] = _real

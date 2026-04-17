# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('cascade.foo') patches
# archivist.lifecycle.cascade.foo (same module object).
import sys

import archivist.lifecycle.cascade as _real

sys.modules[__name__] = _real

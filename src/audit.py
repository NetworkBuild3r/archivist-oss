# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('audit.foo') patches
# archivist.core.audit.foo (same module object).
import sys

import archivist.core.audit as _real

sys.modules[__name__] = _real

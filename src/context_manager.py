# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('context_manager.foo') patches
# archivist.utils.context_manager.foo (same module object).
import sys

import archivist.utils.context_manager as _real

sys.modules[__name__] = _real

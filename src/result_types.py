# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('result_types.foo') patches
# archivist.core.result_types.foo (same module object).
import sys

import archivist.core.result_types as _real

sys.modules[__name__] = _real

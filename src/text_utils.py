# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('text_utils.foo') patches
# archivist.utils.text_utils.foo (same module object).
import sys

import archivist.utils.text_utils as _real

sys.modules[__name__] = _real

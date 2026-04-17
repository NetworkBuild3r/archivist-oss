# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('skills.foo') patches
# archivist.features.skills.foo (same module object).
import sys

import archivist.features.skills as _real

sys.modules[__name__] = _real

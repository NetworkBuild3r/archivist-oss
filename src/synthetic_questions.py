# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('synthetic_questions.foo') patches
# archivist.write.synthetic_questions.foo (same module object).
import sys

import archivist.write.synthetic_questions as _real

sys.modules[__name__] = _real

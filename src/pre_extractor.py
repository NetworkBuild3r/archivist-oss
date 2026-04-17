# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('pre_extractor.foo') patches
# archivist.write.pre_extractor.foo (same module object).
import sys

import archivist.write.pre_extractor as _real

sys.modules[__name__] = _real

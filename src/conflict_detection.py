# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('conflict_detection.foo') patches
# archivist.write.conflict_detection.foo (same module object).
import sys
import archivist.write.conflict_detection as _real
sys.modules[__name__] = _real

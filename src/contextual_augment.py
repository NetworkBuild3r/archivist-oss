# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('contextual_augment.foo') patches
# archivist.write.contextual_augment.foo (same module object).
import sys
import archivist.write.contextual_augment as _real
sys.modules[__name__] = _real

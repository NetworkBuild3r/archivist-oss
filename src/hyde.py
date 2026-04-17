# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('hyde.foo') patches
# archivist.write.hyde.foo (same module object).
import sys
import archivist.write.hyde as _real
sys.modules[__name__] = _real

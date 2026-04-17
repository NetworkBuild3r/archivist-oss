# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('trajectory.foo') patches
# archivist.core.trajectory.foo (same module object).
import sys
import archivist.core.trajectory as _real
sys.modules[__name__] = _real

# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('metrics.foo') patches
# archivist.core.metrics.foo (same module object).
import sys
import archivist.core.metrics as _real
sys.modules[__name__] = _real

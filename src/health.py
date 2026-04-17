# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('health.foo') patches
# archivist.core.health.foo (same module object).
import sys
import archivist.core.health as _real
sys.modules[__name__] = _real

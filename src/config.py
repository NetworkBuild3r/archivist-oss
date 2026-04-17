# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('config.foo') patches
# archivist.core.config.foo (same module object).
import sys
import archivist.core.config as _real
sys.modules[__name__] = _real

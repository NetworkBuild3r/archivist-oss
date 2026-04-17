# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('main.foo') patches
# archivist.app.main.foo (same module object).
import sys
import archivist.app.main as _real
sys.modules[__name__] = _real

# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('curator.foo') patches
# archivist.lifecycle.curator.foo (same module object).
import sys
import archivist.lifecycle.curator as _real
sys.modules[__name__] = _real

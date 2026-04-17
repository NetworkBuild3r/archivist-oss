# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('provenance.foo') patches
# archivist.core.provenance.foo (same module object).
import sys
import archivist.core.provenance as _real
sys.modules[__name__] = _real

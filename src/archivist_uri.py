# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('archivist_uri.foo') patches
# archivist.core.archivist_uri.foo (same module object).
import sys
import archivist.core.archivist_uri as _real
sys.modules[__name__] = _real

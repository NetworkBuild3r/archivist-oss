# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('versioning.foo') patches
# archivist.storage.versioning.foo (same module object).
import sys
import archivist.storage.versioning as _real
sys.modules[__name__] = _real

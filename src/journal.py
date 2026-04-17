# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('journal.foo') patches
# archivist.core.journal.foo (same module object).
import sys
import archivist.core.journal as _real
sys.modules[__name__] = _real

# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('compaction.foo') patches
# archivist.write.compaction.foo (same module object).
import sys
import archivist.write.compaction as _real
sys.modules[__name__] = _real

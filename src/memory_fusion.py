# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('memory_fusion.foo') patches
# archivist.retrieval.memory_fusion.foo (same module object).
import sys
import archivist.retrieval.memory_fusion as _real
sys.modules[__name__] = _real

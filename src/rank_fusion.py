# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('rank_fusion.foo') patches
# archivist.retrieval.rank_fusion.foo (same module object).
import sys
import archivist.retrieval.rank_fusion as _real
sys.modules[__name__] = _real

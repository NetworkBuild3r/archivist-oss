# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('llm.foo') patches
# archivist.features.llm.foo (same module object).
import sys
import archivist.features.llm as _real
sys.modules[__name__] = _real

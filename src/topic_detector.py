# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('topic_detector.foo') patches
# archivist.retrieval.topic_detector.foo (same module object).
import sys
import archivist.retrieval.topic_detector as _real
sys.modules[__name__] = _real

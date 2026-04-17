# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('embeddings.foo') patches
# archivist.features.embeddings.foo (same module object).
import sys

import archivist.features.embeddings as _real

sys.modules[__name__] = _real

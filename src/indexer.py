# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('indexer.foo') patches
# archivist.write.indexer.foo (same module object).
import sys

import archivist.write.indexer as _real

sys.modules[__name__] = _real

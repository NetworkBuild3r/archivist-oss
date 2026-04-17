# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('tokenizer.foo') patches
# archivist.utils.tokenizer.foo (same module object).
import sys

import archivist.utils.tokenizer as _real

sys.modules[__name__] = _real

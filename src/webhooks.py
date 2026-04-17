# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('webhooks.foo') patches
# archivist.features.webhooks.foo (same module object).
import sys

import archivist.features.webhooks as _real

sys.modules[__name__] = _real

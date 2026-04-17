# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('latency_budget.foo') patches
# archivist.core.latency_budget.foo (same module object).
import sys

import archivist.core.latency_budget as _real

sys.modules[__name__] = _real

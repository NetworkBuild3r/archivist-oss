# Compatibility shim — remove in Phase 5 after all imports are updated.
# sys.modules aliasing ensures mock.patch('namespace_inventory.foo') patches
# archivist.storage.namespace_inventory.foo (same module object).
import sys
import archivist.storage.namespace_inventory as _real
sys.modules[__name__] = _real

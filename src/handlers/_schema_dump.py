# Compatibility shim — remove in Phase 5 after all imports are updated.
import sys
import archivist.app.handlers._schema_dump as _real
sys.modules[__name__] = _real

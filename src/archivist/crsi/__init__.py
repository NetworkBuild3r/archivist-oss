"""Continual Recursive Self-Improvement (CRSI) harness loop and Self-Improvement Memory (SIM).

Decoupled subpackage with zero runtime imports from core ``archivist.*`` modules
or ``qdrant_client`` (honors ADR-006 / GR-LAYER-001).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

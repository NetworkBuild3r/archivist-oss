"""RBAC middleware for memory namespace access control.

Uses a namespaces.yaml config file to define per-namespace read/write ACLs.
Falls back to permissive mode if the config is missing (safe default for
single-user / development deployments).
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import yaml

from archivist.core.config import NAMESPACES_CONFIG_PATH, TEAM_MAP

logger = logging.getLogger("archivist.rbac")


@dataclass
class NamespaceConfig:
    id: str
    read: list[str]
    write: list[str]
    consistency: str = "eventual"
    ttl_days: int | None = None
    profile: str = "standard"
    chunk_size: int | None = None
    latency_budget_ms: int | None = None


@dataclass
class AccessPolicy:
    allowed: bool
    reason: str = ""
    hint: str | None = None
    next_steps: list[str] = field(default_factory=list)
    similar_namespaces: list[str] = field(default_factory=list)


@dataclass
class RBACConfig:
    namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)
    agent_namespaces: dict[str, str] = field(default_factory=dict)


_config: RBACConfig | None = None
_permissive_fallback = False


def load_config(path: str = NAMESPACES_CONFIG_PATH) -> RBACConfig:
    """Load namespace + RBAC config from YAML. Falls back to permissive mode on error."""
    global _config, _permissive_fallback

    if not (path or "").strip():
        logger.info(
            "RBAC: NAMESPACES_CONFIG_PATH not set — permissive mode (no namespace ACL file)",
        )
        _permissive_fallback = True
        _config = RBACConfig()
        return _config

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        logger.warning(
            "Failed to load namespace config from %s: %s — FALLING BACK TO PERMISSIVE MODE",
            path,
            e,
        )
        _permissive_fallback = True
        _config = RBACConfig()
        return _config

    _PROFILE_DEFAULTS = {
        "lite": {"chunk_size": 400, "latency_budget_ms": 200},
        "standard": {"chunk_size": 800, "latency_budget_ms": 500},
        "heavy": {"chunk_size": 1200, "latency_budget_ms": 800},
    }

    _permissive_fallback = False
    ns_map: dict[str, NamespaceConfig] = {}
    for ns_raw in raw.get("namespaces", []):
        profile = ns_raw.get("profile", "standard")
        profile_defaults = _PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["standard"])
        ns = NamespaceConfig(
            id=ns_raw["id"],
            read=ns_raw.get("read", []),
            write=ns_raw.get("write", []),
            consistency=ns_raw.get("consistency", "eventual"),
            ttl_days=ns_raw.get("ttl_days"),
            profile=profile,
            chunk_size=ns_raw.get("chunk_size", profile_defaults.get("chunk_size")),
            latency_budget_ms=ns_raw.get(
                "latency_budget_ms", profile_defaults.get("latency_budget_ms")
            ),
        )
        ns_map[ns.id] = ns

    agent_ns = raw.get("agent_namespaces", {})

    _config = RBACConfig(namespaces=ns_map, agent_namespaces=agent_ns)
    logger.info(
        "Loaded RBAC config: %d namespaces, %d agent mappings",
        len(ns_map),
        len(agent_ns),
    )
    return _config


def get_config() -> RBACConfig:
    global _config
    if _config is None:
        load_config()
    return _config


def get_namespace_for_agent(agent_id: str) -> str:
    """Resolve the default namespace for an agent.

    Priority: namespaces.yaml agent_namespaces → TEAM_MAP → 'default'.
    """
    cfg = get_config()
    if agent_id in cfg.agent_namespaces:
        return cfg.agent_namespaces[agent_id]
    team = TEAM_MAP.get(agent_id, "default")
    return team


def get_namespace_config(namespace: str) -> NamespaceConfig | None:
    cfg = get_config()
    return cfg.namespaces.get(namespace)


def _similar_namespaces(candidate: str, known: list[str], top_n: int = 3) -> list[str]:
    """Return up to *top_n* known namespaces that are lexically close to *candidate*.

    Uses SequenceMatcher ratio (0–1).  Threshold 0.4 surfaces obvious typos and
    prefix-mismatches (e.g. "athena-identity" → "athena-identities", "agent-nova"
    → "agents-nova") without returning completely unrelated names.
    """
    scored = [(SequenceMatcher(None, candidate.lower(), ns.lower()).ratio(), ns) for ns in known]
    scored.sort(reverse=True)
    return [ns for ratio, ns in scored if ratio >= 0.4][:top_n]


# Standard next-step recommendations appended to every denial response.
# Agents that follow these will self-correct without human intervention.
_NEXT_STEPS_UNKNOWN_NS = [
    "Call archivist_index(agent_id='<your_agent_id>') to list all namespaces "
    "and memories you currently have access to.",
    "Call archivist_namespaces(agent_id='<your_agent_id>') for a quick namespace "
    "access summary without full memory listing.",
    "Call archivist_get_reference_docs(section='storage') for the complete "
    "archivist_store parameter reference including valid namespace formats.",
]

_NEXT_STEPS_NO_PERMISSION = [
    "Call archivist_namespaces(agent_id='<your_agent_id>') to see which "
    "namespaces you can read and write.",
    "Call archivist_index(agent_id='<your_agent_id>') to browse your accessible "
    "namespaces and their current memory counts.",
    "Call archivist_get_reference_docs(section='admin') for namespace access "
    "configuration guidance.",
]

_NEXT_STEPS_MISSING_CALLER = [
    "Re-issue the call with agent_id='<your_agent_id>' (your unique identifier).",
    "Call archivist_get_reference_docs() to see the full parameter reference for "
    "every tool including required fields.",
]


def check_access(agent_id: str, action: str, namespace: str) -> AccessPolicy:
    """Check if agent_id has permission for action (read/write/delete) on namespace.

    Returns an AccessPolicy with ``allowed=False``, a human-readable ``reason``,
    a short ``hint``, ``next_steps`` (ordered list of actionable calls the agent
    should make to self-correct), and ``similar_namespaces`` (fuzzy matches to
    the requested namespace) on every denial.
    """
    if _permissive_fallback:
        return AccessPolicy(allowed=True, reason="permissive fallback — config not loaded")

    cfg = get_config()
    ns = cfg.namespaces.get(namespace)
    if ns is None:
        all_ns = list(cfg.namespaces.keys())
        similar = _similar_namespaces(namespace, all_ns)
        accessible = list_accessible_namespaces(agent_id)
        accessible_names = [entry["namespace"] for entry in accessible]

        _MAX_HINT_NS = 6
        if len(accessible_names) > _MAX_HINT_NS:
            ns_list = ", ".join(accessible_names[:_MAX_HINT_NS]) + ", ..."
        elif accessible_names:
            ns_list = ", ".join(accessible_names)
        else:
            ns_list = "(none configured — run archivist_index to discover available namespaces)"

        hint_parts = [f"Namespace '{namespace}' does not exist."]
        if similar:
            hint_parts.append(f"Did you mean: {', '.join(similar)}?")
        hint_parts.append(f"Namespaces accessible to '{agent_id}': {ns_list}.")

        return AccessPolicy(
            allowed=False,
            reason=f"Unknown namespace: {namespace}",
            hint=" ".join(hint_parts),
            next_steps=_NEXT_STEPS_UNKNOWN_NS,
            similar_namespaces=similar,
        )

    if action in ("write", "delete"):
        allowed_list = ns.write
    else:
        allowed_list = ns.read

    if "all" in allowed_list:
        return AccessPolicy(allowed=True)

    if agent_id in allowed_list:
        return AccessPolicy(allowed=True)

    action_label = "write to" if action in ("write", "delete") else "read from"
    return AccessPolicy(
        allowed=False,
        reason=f"Agent '{agent_id}' lacks {action} permission for namespace '{namespace}'",
        hint=(
            f"You do not have permission to {action_label} '{namespace}'. "
            f"Call archivist_namespaces(agent_id='{agent_id}') to see your "
            f"accessible namespaces, or archivist_index(agent_id='{agent_id}') "
            f"for a full inventory."
        ),
        next_steps=_NEXT_STEPS_NO_PERMISSION,
        similar_namespaces=[],
    )


def filter_agents_for_read(
    caller_agent_id: str, target_agent_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Return (allowed, denied) target agent IDs for memory read.

    Caller must have read access to each target agent's default namespace.
    """
    allowed: list[str] = []
    denied: list[str] = []
    for tid in target_agent_ids:
        if not tid:
            continue
        ns = get_namespace_for_agent(tid)
        pol = check_access(caller_agent_id, "read", ns)
        if pol.allowed:
            allowed.append(tid)
        else:
            denied.append(tid)
    return allowed, denied


def is_permissive_mode() -> bool:
    """True when namespaces.yaml failed to load and all access is allowed."""
    return _permissive_fallback


def can_read_agent_memory(caller_agent_id: str, target_agent_id: str) -> bool:
    """True if caller may read memories/facts attributed to target_agent_id."""
    if _permissive_fallback:
        return True
    if not target_agent_id:
        return True
    ns = get_namespace_for_agent(target_agent_id)
    return check_access(caller_agent_id, "read", ns).allowed


def list_accessible_namespaces(agent_id: str) -> list[dict]:
    """Return namespaces the agent can access, with read/write flags."""
    cfg = get_config()
    result = []
    for ns_id, ns in cfg.namespaces.items():
        can_read = "all" in ns.read or agent_id in ns.read
        can_write = "all" in ns.write or agent_id in ns.write
        if can_read or can_write:
            result.append(
                {
                    "namespace": ns_id,
                    "can_read": can_read,
                    "can_write": can_write,
                    "consistency": ns.consistency,
                    "ttl_days": ns.ttl_days,
                }
            )
    return result

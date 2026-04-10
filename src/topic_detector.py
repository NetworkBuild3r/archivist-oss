"""Lightweight keyword-based topic classifier for memory chunks and queries.

Zero-latency, no LLM — pure keyword overlap scoring.  Used at index time
to tag chunks and at query time to pre-filter vector search by topic.

The default keyword map covers common DevOps/SRE domains.  Override via a
YAML file pointed to by the ``TOPIC_MAP_PATH`` environment variable.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter

logger = logging.getLogger("archivist.topic_detector")

# ---------------------------------------------------------------------------
# Default topic → keyword map (aligned with benchmark TOPICS)
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "kubernetes": [
        "kubernetes", "k8s", "cluster", "pod", "node", "deployment",
        "service mesh", "ingress", "helm", "karpenter", "hpa",
        "namespace", "kubelet", "kube-proxy", "istio", "kubectl",
    ],
    "cicd": [
        "pipeline", "ci/cd", "cicd", "gitlab", "deploy", "build",
        "release", "argocd", "argo cd", "semantic versioning",
        "changelog", "docker", "container image", "registry",
    ],
    "monitoring": [
        "prometheus", "grafana", "alert", "metric", "dashboard",
        "slo", "sli", "sla", "pagerduty", "thanos", "observability",
        "tracing", "jaeger", "opentelemetry", "loki", "logging",
    ],
    "database": [
        "database", "postgres", "postgresql", "mysql", "migration",
        "schema", "query", "index", "rds", "pgbouncer", "flyway",
        "redis", "dynamodb", "mongodb", "sql", "replication",
    ],
    "security": [
        "security", "auth", "authentication", "authorization", "rbac",
        "certificate", "vault", "encryption", "tls", "ssl", "jwt",
        "opa", "policy", "kms", "secret", "credential",
    ],
    "incident": [
        "incident", "outage", "rollback", "postmortem", "recovery",
        "mttr", "downtime", "degradation", "on-call", "escalation",
        "root cause", "rca", "severity",
    ],
    "architecture": [
        "microservice", "api", "gateway", "queue", "event",
        "rabbitmq", "kafka", "grpc", "protobuf", "etcd",
        "service mesh", "load balancer", "cdn",
    ],
    "testing": [
        "test", "coverage", "integration test", "e2e", "unit test",
        "fixture", "playwright", "pytest", "flaky", "regression",
    ],
    "performance": [
        "latency", "throughput", "cache", "optimization", "scaling",
        "auto-scaling", "benchmark", "p99", "p50", "qps",
    ],
    "networking": [
        "dns", "ip", "subnet", "vpn", "tailscale", "wireguard",
        "firewall", "proxy", "route", "bgp", "vlan", "cidr",
    ],
}

# Compiled regex patterns per topic (built once at import time or on reload)
_compiled: dict[str, list[re.Pattern]] = {}


def _compile_patterns():
    """Build case-insensitive regex patterns from keyword lists."""
    global _compiled
    _compiled = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        patterns = []
        for kw in keywords:
            escaped = re.escape(kw)
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        _compiled[topic] = patterns


def _load_custom_topic_map():
    """Override default keywords from a YAML file if TOPIC_MAP_PATH is set."""
    path = os.getenv("TOPIC_MAP_PATH", "")
    if not path or not os.path.isfile(path):
        return
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            for topic, keywords in data.items():
                if isinstance(keywords, list):
                    TOPIC_KEYWORDS[str(topic)] = [str(k) for k in keywords]
            logger.info("Loaded custom topic map from %s (%d topics)", path, len(data))
    except Exception as e:
        logger.warning("Failed to load topic map from %s: %s", path, e)


_load_custom_topic_map()
_compile_patterns()


def detect_topics(text: str, top_n: int = 2) -> list[str]:
    """Return top-N topic labels for a text chunk, ranked by keyword hits.

    Returns an empty list if no topic exceeds the minimum threshold (2 hits).
    """
    if not text:
        return []

    counts: Counter = Counter()
    for topic, patterns in _compiled.items():
        for pat in patterns:
            hits = len(pat.findall(text))
            if hits:
                counts[topic] += hits

    if not counts:
        return []

    min_hits = 2
    ranked = [(t, c) for t, c in counts.most_common() if c >= min_hits]
    return [t for t, _ in ranked[:top_n]]


def detect_query_topic(query: str) -> str:
    """Return the single most likely topic for a search query, or '' if unclear.

    Uses a lower threshold (1 hit) since queries are short.
    """
    if not query:
        return ""

    counts: Counter = Counter()
    for topic, patterns in _compiled.items():
        for pat in patterns:
            hits = len(pat.findall(query))
            if hits:
                counts[topic] += hits

    if not counts:
        return ""

    best_topic, _best_count = counts.most_common(1)[0]
    return best_topic

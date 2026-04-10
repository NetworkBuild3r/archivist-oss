"""Generate seed corpus and questions.json.

Legacy (50 files → benchmarks/fixtures/corpus/):
    python benchmarks/fixtures/generate_corpus.py

Dual-track scale presets (Phase 2):
    python benchmarks/fixtures/generate_corpus.py --preset small
    python benchmarks/fixtures/generate_corpus.py --preset medium
    python benchmarks/fixtures/generate_corpus.py --preset large
    python benchmarks/fixtures/generate_corpus.py --preset stress   # huge haystack; same questions as large

Questions file (adds needle / contradiction tags; use with any preset):
    python benchmarks/fixtures/generate_corpus.py --write-questions
"""

import argparse
import json
import os
import random
import re
from datetime import date, timedelta

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")

# Needle facts for large/medium corpora (unique strings for haystack tests)
NEEDLE_SECRET = (
    "ArchivistNeedleV1: the approved production backup window is exactly 04:15 UTC every Sunday; "
    "no other window is authorized."
)
NEEDLE_NUMERIC = "The fleet-wide memory budget ceiling is exactly 47,392 MiB per node, enforced by cgroup v2."
NEEDLE_NAME = "The on-call DBA for Q2 2026 is Dr. Ramona Vex (employee ID RV-8834), reachable at +1-555-0142."
NEEDLE_CONFIG = "ARCHIVIST_MAX_VECTOR_BATCH=256 ARCHIVIST_REINDEX_CRON='15 3 * * 6' ARCHIVIST_SHARD_KEY=murmur3"
NEEDLE_IP = "The canary egress gateway is 198.51.100.73:8443 with TLS fingerprint sha256:a1b2c3d4e5f6."
NEEDLE_CROSS_AGENT = "Project Obsidian launch date is 2026-07-14; budget approved at $2.4M by CFO on 2026-03-01."
CONTRADICTION_FACT_A = (
    "Monitoring SLO (authoritative Feb 2026): API p99 latency must stay under 200ms at all times."
)
CONTRADICTION_FACT_B = (
    "Monitoring SLO (revised note): API p99 latency target was relaxed to 500ms for edge regions."
)

PRESETS = {
    "small": {
        "agents": 6,
        "files_per_agent": 9,
        "noise_files": 0,
        "days_span": 75,
        "needle": False,
        "paraphrase_dupes": 0,
        "contradiction_files": True,
    },
    "medium": {
        "agents": 10,
        "files_per_agent": 10,
        "noise_files": 120,
        "days_span": 120,
        "needle": True,
        "paraphrase_dupes": 30,
        "contradiction_files": True,
    },
    "large": {
        "agents": 10,
        "files_per_agent": 14,
        "noise_files": 320,
        "days_span": 150,
        "needle": True,
        "paraphrase_dupes": 80,
        "contradiction_files": True,
    },
    # Very large haystack: same question set as `large` in evaluate.py, extra noise + span for "memory not lost" demos.
    "stress": {
        "agents": 10,
        "files_per_agent": 20,
        "noise_files": 1200,
        "days_span": 400,
        "needle": True,
        "paraphrase_dupes": 120,
        "contradiction_files": True,
    },
}

AGENTS = ["chief", "gitbob", "grafgreg", "argo", "kubekate",
           "devdan", "securitysam", "docbot", "testrunner", "deployer"]

NAMESPACES = ["pipeline", "deployer", "shared", "chief", "monitoring"]

TOPICS = {
    "kubernetes": {
        "keywords": ["kubernetes", "k8s", "cluster", "pod", "deployment", "service", "ingress"],
        "facts": [
            "The production Kubernetes cluster runs v1.29 across 12 nodes in us-east-1.",
            "Pod autoscaling is configured with HPA targeting 70% CPU utilization.",
            "The staging cluster uses spot instances to reduce cost by 60%.",
            "Istio service mesh is enabled for mTLS between all services.",
            "Node pool scaling uses Karpenter with provisioner limits of 100 CPUs.",
        ],
    },
    "cicd": {
        "keywords": ["pipeline", "CI/CD", "GitLab", "deploy", "build", "release"],
        "facts": [
            "The CI pipeline runs on GitLab CI with a 15-minute timeout per job.",
            "Deployment uses ArgoCD with automated sync enabled for staging.",
            "Production deployments require manual approval from the chief agent.",
            "The build pipeline caches Docker layers to reduce build time to under 5 minutes.",
            "Release tags follow semantic versioning with automatic changelog generation.",
        ],
    },
    "monitoring": {
        "keywords": ["prometheus", "grafana", "alert", "metric", "dashboard", "SLO"],
        "facts": [
            "Prometheus scrapes metrics every 15 seconds from all services.",
            "The SLO for API latency is p99 under 500ms with 99.9% availability.",
            "Grafana dashboards are provisioned from code stored in the infra repo.",
            "PagerDuty integration fires alerts when error rate exceeds 1% for 5 minutes.",
            "The monitoring stack uses Thanos for long-term metric storage with 90-day retention.",
        ],
    },
    "database": {
        "keywords": ["database", "postgres", "migration", "schema", "query", "index"],
        "facts": [
            "The primary database is PostgreSQL 16 running on RDS with Multi-AZ.",
            "Schema migrations use Flyway with a separate approval process for production.",
            "Read replicas handle reporting queries to avoid impacting write throughput.",
            "The users table has a GIN index on the metadata JSONB column.",
            "Connection pooling uses PgBouncer with a max of 200 connections per service.",
        ],
    },
    "security": {
        "keywords": ["security", "auth", "RBAC", "certificate", "vault", "encryption"],
        "facts": [
            "All secrets are stored in HashiCorp Vault with auto-rotation every 30 days.",
            "JWT tokens expire after 1 hour with refresh tokens valid for 7 days.",
            "TLS certificates are managed by cert-manager with Let's Encrypt.",
            "RBAC policies use OPA (Open Policy Agent) for fine-grained authorization.",
            "All data at rest is encrypted with AES-256 using AWS KMS customer-managed keys.",
        ],
    },
    "architecture": {
        "keywords": ["microservice", "API", "gateway", "queue", "event", "service"],
        "facts": [
            "The system uses an event-driven architecture with RabbitMQ as the message broker.",
            "The API gateway rate-limits at 1000 requests per second per API key.",
            "Service-to-service communication uses gRPC with protobuf contracts.",
            "The order processing service publishes events to three downstream consumers.",
            "The configuration service uses etcd for distributed configuration management.",
        ],
    },
    "testing": {
        "keywords": ["test", "coverage", "integration", "e2e", "unit", "fixture"],
        "facts": [
            "Unit test coverage is maintained above 85% across all services.",
            "Integration tests run against a Docker Compose environment nightly.",
            "E2E tests use Playwright and run in parallel across 4 browser types.",
            "The test suite completes in under 10 minutes for the full pipeline.",
            "Flaky test detection uses a quarantine system that retries 3 times before failing.",
        ],
    },
    "incident": {
        "keywords": ["incident", "outage", "rollback", "postmortem", "recovery"],
        "facts": [
            "On 2025-02-15, a DNS misconfiguration caused 45 minutes of downtime.",
            "The rollback procedure takes under 3 minutes using ArgoCD's sync revert.",
            "Incident response follows a blameless postmortem template within 48 hours.",
            "The MTTR target is under 30 minutes for P1 incidents.",
            "Automatic rollback triggers when the error rate exceeds 5% within 2 minutes of deployment.",
        ],
    },
    "performance": {
        "keywords": ["latency", "throughput", "cache", "optimization", "scaling"],
        "facts": [
            "Redis cache reduces API response time from 200ms to 15ms for hot paths.",
            "The search service handles 5000 queries per second at p50 latency of 30ms.",
            "CDN caching serves 80% of static assets without hitting the origin server.",
            "Database query optimization reduced the daily reporting job from 4 hours to 20 minutes.",
            "Auto-scaling adds new instances within 90 seconds when CPU exceeds 75%.",
        ],
    },
    "cost": {
        "keywords": ["cost", "budget", "savings", "optimization", "billing"],
        "facts": [
            "Monthly cloud spend is $47,000 with a target to reduce to $40,000 by Q3.",
            "Switching to ARM-based instances saved 25% on compute costs.",
            "Reserved instances for the database tier saved $8,000 annually.",
            "The data transfer cost between regions is $2,100 per month.",
            "Spot instances for batch processing reduced costs by 70% compared to on-demand.",
        ],
    },
}


def _generate_file(agent: str, topic_name: str, topic: dict, date: str, file_idx: int) -> str:
    """Generate a single markdown file content."""
    facts = random.sample(topic["facts"], min(3, len(topic["facts"])))
    related_topic = random.choice(list(TOPICS.keys()))
    related_facts = random.sample(TOPICS[related_topic]["facts"], min(2, len(TOPICS[related_topic]["facts"])))

    lines = [
        f"# {topic_name.title()} Update — {date}",
        f"",
        f"**Agent:** {agent}",
        f"**Date:** {date}",
        f"**Topic:** {topic_name}",
        f"",
        f"## Summary",
        f"",
    ]

    for fact in facts:
        lines.append(f"- {fact}")
    lines.append("")

    lines.append(f"## Related: {related_topic.title()}")
    lines.append("")
    for fact in related_facts:
        lines.append(f"- {fact}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"This was recorded by {agent} during routine operations. "
        f"Cross-referenced with {random.choice(AGENTS)} observations."
    )
    lines.append("")

    if file_idx % 5 == 0:
        lines.append("## Contradiction Alert")
        lines.append("")
        lines.append(
            f"Note: {random.choice(AGENTS)} reported conflicting information "
            f"about {random.choice(topic['keywords'])} on {date}. "
            f"This may require verification."
        )
        lines.append("")

    return "\n".join(lines)


def _generate_questions() -> list[dict]:
    """Generate 100 question/answer pairs spanning all topics and query types."""
    questions = []
    qid = 0

    for topic_name, topic in TOPICS.items():
        for fact in topic["facts"]:
            qid += 1
            questions.append({
                "id": qid,
                "query": _fact_to_question(fact, topic_name),
                "expected_keywords": _extract_keywords(fact),
                "expected_answer": fact,
                "topic": topic_name,
                "query_type": "single_hop",
                "difficulty": "easy",
            })

    cross_topic_pairs = [
        ("kubernetes", "monitoring", "How does the monitoring stack observe the Kubernetes cluster?"),
        ("cicd", "security", "What security measures protect the CI/CD pipeline?"),
        ("database", "performance", "How does database optimization affect overall system performance?"),
        ("architecture", "testing", "How are microservices tested in the integration environment?"),
        ("incident", "monitoring", "How do alerts relate to incident response?"),
        ("cost", "kubernetes", "What cost optimizations exist for Kubernetes compute?"),
        ("security", "database", "How is database access controlled and encrypted?"),
        ("cicd", "kubernetes", "How does the CI/CD pipeline deploy to Kubernetes?"),
        ("performance", "architecture", "How does the API gateway handle high throughput?"),
        ("testing", "cicd", "How are tests integrated into the CI pipeline?"),
    ]

    for t1, t2, query in cross_topic_pairs:
        qid += 1
        keywords = TOPICS[t1]["keywords"][:2] + TOPICS[t2]["keywords"][:2]
        expected = f"{TOPICS[t1]['facts'][0]} Additionally, {TOPICS[t2]['facts'][0]}"
        questions.append({
            "id": qid,
            "query": query,
            "expected_keywords": keywords,
            "expected_answer": expected,
            "topic": f"{t1}+{t2}",
            "query_type": "multi_hop",
            "difficulty": "medium",
        })

    temporal_questions = [
        {"query": "What happened on 2025-02-15?",
         "expected_keywords": ["DNS", "misconfiguration", "downtime", "45 minutes"],
         "expected_answer": "On 2025-02-15, a DNS misconfiguration caused 45 minutes of downtime.",
         "topic": "incident", "query_type": "temporal", "difficulty": "easy",
         "date_from": "2025-02-15", "date_to": "2025-02-15"},
        {"query": "What are the most recent changes to the deployment pipeline?",
         "expected_keywords": ["ArgoCD", "staging", "automated", "sync"],
         "expected_answer": "Deployment uses ArgoCD with automated sync enabled for staging.",
         "topic": "cicd", "query_type": "temporal", "difficulty": "medium",
         "namespace": "pipeline"},
        {"query": "What is the current state of production Kubernetes?",
         "expected_keywords": ["v1.29", "12 nodes", "us-east-1"],
         "expected_answer": "The production Kubernetes cluster runs v1.29 across 12 nodes in us-east-1.",
         "topic": "kubernetes", "query_type": "temporal", "difficulty": "easy"},
        {"query": "When was the last significant incident?",
         "expected_keywords": ["2025-02-15", "DNS", "downtime"],
         "expected_answer": "On 2025-02-15, a DNS misconfiguration caused 45 minutes of downtime.",
         "topic": "incident", "query_type": "temporal", "difficulty": "medium"},
        {"query": "What security changes have been made recently?",
         "expected_keywords": ["Vault", "rotation", "30 days", "TLS", "cert-manager"],
         "expected_answer": "All secrets are stored in HashiCorp Vault with auto-rotation every 30 days.",
         "topic": "security", "query_type": "temporal", "difficulty": "medium"},
    ]
    for tq in temporal_questions:
        qid += 1
        questions.append({"id": qid, **tq})

    adversarial_questions = [
        {"query": "What is the Redis cache hit rate for the users table?",
         "expected_keywords": [],
         "expected_answer": "",
         "topic": "performance+database", "query_type": "adversarial", "difficulty": "hard"},
        {"query": "How many pods are running in the staging cluster right now?",
         "expected_keywords": [],
         "expected_answer": "",
         "topic": "kubernetes", "query_type": "adversarial", "difficulty": "hard"},
        {"query": "What is the total number of microservices in production?",
         "expected_keywords": [],
         "expected_answer": "",
         "topic": "architecture", "query_type": "adversarial", "difficulty": "hard"},
        {"query": "Who approved the last production deployment?",
         "expected_keywords": ["chief", "manual approval"],
         "expected_answer": "Production deployments require manual approval from the chief agent.",
         "topic": "cicd", "query_type": "adversarial", "difficulty": "hard"},
        {"query": "What are the conflicting reports about Kubernetes from different agents?",
         "expected_keywords": ["conflicting", "contradiction"],
         "expected_answer": "",
         "topic": "kubernetes", "query_type": "adversarial", "difficulty": "hard"},
    ]
    for aq in adversarial_questions:
        qid += 1
        questions.append({"id": qid, **aq})

    agent_scope_questions = [
        {"query": "What has the chief agent reported about costs?",
         "expected_keywords": ["$47,000", "budget", "cloud spend"],
         "expected_answer": "Monthly cloud spend is $47,000 with a target to reduce to $40,000 by Q3.",
         "topic": "cost", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "chief", "caller_agent_id": "chief", "namespace": "chief"},
        {"query": "What does gitbob know about the CI pipeline?",
         "expected_keywords": ["GitLab CI", "15-minute", "timeout"],
         "expected_answer": "The CI pipeline runs on GitLab CI with a 15-minute timeout per job.",
         "topic": "cicd", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "gitbob", "caller_agent_id": "gitbob", "namespace": "pipeline"},
        {"query": "What monitoring information has grafgreg recorded?",
         "expected_keywords": ["Prometheus", "Grafana", "SLO", "dashboard"],
         "expected_answer": "Prometheus scrapes metrics every 15 seconds from all services.",
         "topic": "monitoring", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "grafgreg", "caller_agent_id": "grafgreg", "namespace": "monitoring"},
        {"query": "What deployments has argo managed?",
         "expected_keywords": ["ArgoCD", "sync", "staging", "rollback"],
         "expected_answer": "Deployment uses ArgoCD with automated sync enabled for staging.",
         "topic": "cicd", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "argo", "caller_agent_id": "argo", "namespace": "deployer"},
        {"query": "What has securitysam flagged about authentication?",
         "expected_keywords": ["JWT", "Vault", "RBAC", "OPA"],
         "expected_answer": "JWT tokens expire after 1 hour with refresh tokens valid for 7 days.",
         "topic": "security", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "securitysam", "caller_agent_id": "securitysam", "namespace": "shared"},
    ]
    for sq in agent_scope_questions:
        qid += 1
        questions.append({"id": qid, **sq})

    contradiction_questions = [
        {
            "query": "What are the stated API p99 latency SLO values for monitoring, including conflicting notes?",
            "expected_keywords": ["200ms", "500ms", "p99", "latency", "SLO"],
            "expected_answer": CONTRADICTION_FACT_A,
            "topic": "monitoring",
            "query_type": "contradiction",
            "tags": ["contradiction", "multi_hop"],
            "scales": ["small", "medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "Is the monitoring latency target 200ms or 500ms p99?",
            "expected_keywords": ["200", "500", "p99"],
            "expected_answer": CONTRADICTION_FACT_B,
            "topic": "monitoring",
            "query_type": "contradiction",
            "tags": ["contradiction"],
            "scales": ["small", "medium", "large"],
            "difficulty": "hard",
        },
    ]
    for cq in contradiction_questions:
        qid += 1
        questions.append({"id": qid, **cq})

    needle_questions = [
        # ── Original backup window needle (3 variants) ──
        {
            "query": "What is the exact approved production backup window (time and day)?",
            "expected_keywords": ["ArchivistNeedleV1", "04:15", "Sunday", "UTC"],
            "expected_answer": NEEDLE_SECRET,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "single_hop"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "Find the canonical backup schedule phrase used for production.",
            "expected_keywords": ["ArchivistNeedleV1", "backup"],
            "expected_answer": NEEDLE_SECRET,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "needleagent backup policy UTC",
            "expected_keywords": ["04:15", "Sunday"],
            "expected_answer": NEEDLE_SECRET,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle"],
            "scales": ["large"],
            "difficulty": "medium",
        },
        # ── Paraphrased needle (different words, same target) ──
        {
            "query": "When is the scheduled maintenance window for production systems?",
            "expected_keywords": ["04:15", "Sunday"],
            "expected_answer": NEEDLE_SECRET,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "paraphrase"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "What day and time are backups authorized to run?",
            "expected_keywords": ["04:15", "Sunday", "UTC"],
            "expected_answer": NEEDLE_SECRET,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "paraphrase"],
            "scales": ["large"],
            "difficulty": "hard",
        },
        # ── Numeric needle ──
        {
            "query": "What is the per-node memory budget ceiling in MiB?",
            "expected_keywords": ["47,392", "MiB", "cgroup"],
            "expected_answer": NEEDLE_NUMERIC,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "numeric"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "fleet memory limit per node",
            "expected_keywords": ["47,392", "MiB"],
            "expected_answer": NEEDLE_NUMERIC,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "numeric"],
            "scales": ["large"],
            "difficulty": "medium",
        },
        # ── Name needle (person/contact) ──
        {
            "query": "Who is the on-call DBA for Q2 2026 and what is their contact number?",
            "expected_keywords": ["Ramona Vex", "RV-8834", "555-0142"],
            "expected_answer": NEEDLE_NAME,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "name"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "DBA on-call rotation Q2",
            "expected_keywords": ["Ramona Vex"],
            "expected_answer": NEEDLE_NAME,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "name"],
            "scales": ["large"],
            "difficulty": "medium",
        },
        # ── Configuration needle (env vars/settings) ──
        {
            "query": "What is the ARCHIVIST_MAX_VECTOR_BATCH setting and the reindex cron schedule?",
            "expected_keywords": ["256", "15 3 * * 6", "murmur3"],
            "expected_answer": NEEDLE_CONFIG,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "config"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "archivist reindex cron schedule production",
            "expected_keywords": ["15 3 * * 6"],
            "expected_answer": NEEDLE_CONFIG,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "config"],
            "scales": ["large"],
            "difficulty": "medium",
        },
        # ── IP/network needle ──
        {
            "query": "What is the canary egress gateway IP and port?",
            "expected_keywords": ["198.51.100.73", "8443"],
            "expected_answer": NEEDLE_IP,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "network"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "canary gateway TLS fingerprint",
            "expected_keywords": ["198.51.100.73", "a1b2c3d4e5f6"],
            "expected_answer": NEEDLE_IP,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "network"],
            "scales": ["large"],
            "difficulty": "hard",
        },
        # ── Cross-agent needle (stored by ops-planning, queried generically) ──
        {
            "query": "When is Project Obsidian launching and what is the approved budget?",
            "expected_keywords": ["2026-07-14", "$2.4M"],
            "expected_answer": NEEDLE_CROSS_AGENT,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "cross_agent"],
            "scales": ["medium", "large"],
            "difficulty": "hard",
        },
        {
            "query": "Project Obsidian launch date",
            "expected_keywords": ["2026-07-14"],
            "expected_answer": NEEDLE_CROSS_AGENT,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "cross_agent"],
            "scales": ["large"],
            "difficulty": "medium",
        },
        # ── Temporal needle (specific date context) ──
        {
            "query": "What capacity planning decision was made on 2026-02-11?",
            "expected_keywords": ["47,392", "MiB"],
            "expected_answer": NEEDLE_NUMERIC,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "temporal"],
            "scales": ["large"],
            "difficulty": "hard",
        },
        {
            "query": "What network configuration was documented on March 18, 2026?",
            "expected_keywords": ["198.51.100.73", "canary"],
            "expected_answer": NEEDLE_IP,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "temporal"],
            "scales": ["large"],
            "difficulty": "hard",
        },
        {
            "query": "Who was assigned as DBA contact in late February 2026?",
            "expected_keywords": ["Ramona Vex"],
            "expected_answer": NEEDLE_NAME,
            "topic": "operations",
            "query_type": "needle",
            "tags": ["needle", "temporal", "name"],
            "scales": ["large"],
            "difficulty": "hard",
        },
    ]
    for nq in needle_questions:
        qid += 1
        questions.append({"id": qid, **nq})

    while len(questions) < 110:
        topic_name = random.choice(list(TOPICS.keys()))
        topic = TOPICS[topic_name]
        fact = random.choice(topic["facts"])
        qid += 1
        questions.append({
            "id": qid,
            "query": f"Tell me about {random.choice(topic['keywords'])}",
            "expected_keywords": _extract_keywords(fact),
            "expected_answer": fact,
            "topic": topic_name,
            "query_type": "broad",
            "difficulty": "easy",
        })

    for q in questions:
        q.setdefault("tags", [q.get("query_type", "single_hop")])
        q.setdefault("scales", ["small", "medium", "large"])

    return questions[:120]


def _fact_to_question(fact: str, topic: str) -> str:
    """Convert a fact statement into a natural question."""
    transforms = [
        lambda f, t: f"What do we know about {random.choice(TOPICS[t]['keywords'])}?",
        lambda f, t: f"Describe the {t} configuration.",
        lambda f, t: f"What is the current {t} setup?",
        lambda f, t: f"Tell me the details about {random.choice(TOPICS[t]['keywords'])}.",
    ]
    return random.choice(transforms)(fact, topic)


def _extract_keywords(fact: str) -> list[str]:
    """Pull key terms from a fact for keyword matching evaluation."""
    numbers = re.findall(r'\d+[\d,.]*%?', fact)
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', fact)
    return (numbers + proper_nouns)[:5]


def _spread_dates(n: int, days_span: int) -> list[str]:
    """Return n ISO dates spread across ~days_span days ending near 2026-03-30."""
    end = date(2026, 3, 30)
    start = end - timedelta(days=max(days_span, n))
    if n <= 1:
        return [end.isoformat()]
    step = max(1, (end - start).days // max(n - 1, 1))
    out = []
    d = start
    for _ in range(n):
        out.append(d.isoformat())
        d = min(end, d + timedelta(days=step))
    return out


def _write_contradiction_pair(corpus_root: str) -> None:
    chief = os.path.join(corpus_root, "agents", "chief")
    grafgreg = os.path.join(corpus_root, "agents", "grafgreg")
    os.makedirs(chief, exist_ok=True)
    os.makedirs(grafgreg, exist_ok=True)
    with open(os.path.join(chief, "2026-02-01-monitoring-slo.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Monitoring SLO (chief)\n\n**Agent:** chief\n**Date:** 2026-02-01\n\n"
            f"## Fact\n\n{CONTRADICTION_FACT_A}\n\nRelated: Prometheus, Grafana, Kubernetes.\n"
        )
    with open(os.path.join(grafgreg, "2026-02-03-monitoring-slo-edge.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Monitoring SLO edge regions (grafgreg)\n\n**Agent:** grafgreg\n**Date:** 2026-02-03\n\n"
            f"## Update\n\n{CONTRADICTION_FACT_B}\n\nContext: PostgreSQL, ArgoCD, latency budgets.\n"
        )


def _write_needle_file(corpus_root: str) -> None:
    ndir = os.path.join(corpus_root, "agents", "needleagent")
    os.makedirs(ndir, exist_ok=True)
    with open(os.path.join(ndir, "2026-01-20-backup-policy.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Backup policy note\n\n**Agent:** needleagent\n**Date:** 2026-01-20\n\n"
            "## Canonical window\n\n"
            f"{NEEDLE_SECRET}\n\n"
            "Unrelated filler: Kubernetes PostgreSQL ArgoCD monitoring deployment pipeline.\n"
        )
    with open(os.path.join(ndir, "2026-02-11-capacity-limits.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Capacity planning\n\n**Agent:** needleagent\n**Date:** 2026-02-11\n\n"
            f"## Memory budget\n\n{NEEDLE_NUMERIC}\n\n"
            "General notes: disk IOPS, network throughput, CPU scheduling.\n"
        )
    with open(os.path.join(ndir, "2026-02-25-oncall-roster.md"), "w", encoding="utf-8") as f:
        f.write(
            "# On-call roster Q2\n\n**Agent:** needleagent\n**Date:** 2026-02-25\n\n"
            f"## DBA rotation\n\n{NEEDLE_NAME}\n\n"
            "Backup contacts: ops-team@company.com, #incident-room Slack.\n"
        )
    with open(os.path.join(ndir, "2026-03-05-archivist-tuning.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Archivist production tuning\n\n**Agent:** needleagent\n**Date:** 2026-03-05\n\n"
            f"## Env overrides\n\n{NEEDLE_CONFIG}\n\n"
            "Applied after the v1.9 rollout to reduce reindex contention.\n"
        )
    with open(os.path.join(ndir, "2026-03-18-network-edge.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Edge network config\n\n**Agent:** needleagent\n**Date:** 2026-03-18\n\n"
            f"## Canary gateway\n\n{NEEDLE_IP}\n\n"
            "Used for pre-production traffic mirroring from us-west-2.\n"
        )
    # Cross-agent needle: stored by a different agent
    odir = os.path.join(corpus_root, "agents", "ops-planning")
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, "2026-03-01-project-obsidian.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Project Obsidian planning\n\n**Agent:** ops-planning\n**Date:** 2026-03-01\n\n"
            f"## Launch details\n\n{NEEDLE_CROSS_AGENT}\n\n"
            "Stakeholders: engineering, product, finance.\n"
        )


def _write_noise_file(corpus_root: str, idx: int, rng: random.Random) -> None:
    ndir = os.path.join(corpus_root, "agents", "noise_bot")
    os.makedirs(ndir, exist_ok=True)
    tname = rng.choice(list(TOPICS.keys()))
    topic = TOPICS[tname]
    fact = rng.choice(topic["facts"])
    noise = rng.choice(topic["facts"])
    day = 1 + (idx % 28)
    content = (
        f"# Routine log {idx}\n\n**Agent:** noise_bot\n**Date:** 2025-06-{day:02d}\n\n"
        f"- {fact}\n- {noise}\n- Reference: {rng.choice(topic['keywords'])} optimization.\n"
    )
    with open(os.path.join(ndir, f"noise_{idx:05d}.md"), "w", encoding="utf-8") as f:
        f.write(content)


def _write_paraphrase_dupes(corpus_root: str, count: int, rng: random.Random) -> None:
    ndir = os.path.join(corpus_root, "agents", "paraphrase_bot")
    os.makedirs(ndir, exist_ok=True)
    base = "The CI pipeline uses GitLab CI with a fifteen minute timeout cap on each job."
    for i in range(count):
        variants = [
            base,
            "GitLab CI enforces a 15-minute timeout per pipeline job.",
            "Each CI job is limited to fifteen minutes in GitLab CI.",
        ]
        text = rng.choice(variants)
        with open(os.path.join(ndir, f"dup_{i:04d}.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# CI note variant {i}\n\n**Agent:** paraphrase_bot\n**Date:** 2025-08-{(i % 28) + 1:02d}\n\n{text}\n"
            )


def generate_preset_corpus(preset: str, fixtures_dir: str) -> tuple[str, int]:
    """Write corpus files for a scale preset. Returns (corpus_root, file_count)."""
    import shutil

    cfg = PRESETS[preset]
    corpus_root = os.path.join(fixtures_dir, f"corpus_{preset}")
    if os.path.isdir(corpus_root):
        shutil.rmtree(corpus_root)
    os.makedirs(corpus_root, exist_ok=True)

    rng = random.Random(42 + sum(ord(c) for c in preset))
    agents_use = AGENTS[: cfg["agents"]]
    n_agent_files = cfg["agents"] * cfg["files_per_agent"]
    dates = _spread_dates(n_agent_files, cfg["days_span"])
    topic_names = list(TOPICS.keys())
    file_idx = 0

    for agent_idx, agent in enumerate(agents_use):
        agent_dir = os.path.join(corpus_root, "agents", agent)
        os.makedirs(agent_dir, exist_ok=True)
        for doc_idx in range(cfg["files_per_agent"]):
            topic_name = topic_names[(agent_idx + doc_idx) % len(topic_names)]
            topic = TOPICS[topic_name]
            date_str = dates[file_idx % len(dates)]
            content = _generate_file(agent, topic_name, topic, date_str, file_idx)
            safe_name = date_str.replace("-", "") + f"_{doc_idx}.md"
            filepath = os.path.join(agent_dir, safe_name)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            file_idx += 1

    if cfg.get("contradiction_files"):
        _write_contradiction_pair(corpus_root)
        file_idx += 2

    if cfg.get("needle"):
        _write_needle_file(corpus_root)
        file_idx += 1

    for i in range(cfg["noise_files"]):
        _write_noise_file(corpus_root, i, rng)
        file_idx += 1

    if cfg.get("paraphrase_dupes", 0) > 0:
        _write_paraphrase_dupes(corpus_root, cfg["paraphrase_dupes"], rng)
        file_idx += cfg["paraphrase_dupes"]

    return corpus_root, file_idx


def generate_legacy_corpus(corpus_root: str) -> int:
    """Original 50-file layout under corpus/."""
    os.makedirs(corpus_root, exist_ok=True)
    topic_names = list(TOPICS.keys())
    file_idx = 0
    for agent_idx, agent in enumerate(AGENTS):
        agent_dir = os.path.join(corpus_root, "agents", agent)
        os.makedirs(agent_dir, exist_ok=True)
        for doc_idx in range(5):
            topic_name = topic_names[(agent_idx + doc_idx) % len(topic_names)]
            topic = TOPICS[topic_name]
            day = 1 + (agent_idx * 3 + doc_idx) % 28
            date = f"2025-03-{day:02d}"
            content = _generate_file(agent, topic_name, topic, date, file_idx)
            filename = f"{date}.md"
            filepath = os.path.join(agent_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            file_idx += 1
    return file_idx


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark corpus / questions")
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large", "stress"],
        help="Write scaled corpus to benchmarks/fixtures/corpus_<preset>/",
    )
    parser.add_argument(
        "--write-questions",
        action="store_true",
        help="Write benchmarks/fixtures/questions.json (with needle + contradiction entries)",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="With --preset: only write corpus files, not questions",
    )
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Only regenerate questions.json (no corpus files)",
    )
    args = parser.parse_args()
    fixtures_dir = os.path.dirname(__file__)
    random.seed(42)

    def _write_questions_file():
        questions = _generate_questions()
        with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        type_counts = {}
        for q in questions:
            qt = q["query_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1
        print(f"Generated {len(questions)} questions in {QUESTIONS_PATH}")
        print(f"  By type: {type_counts}")

    if args.questions_only:
        _write_questions_file()
        return

    if args.preset:
        root, nfiles = generate_preset_corpus(args.preset, fixtures_dir)
        print(f"Preset {args.preset}: {nfiles} files under {root}")
    else:
        os.makedirs(CORPUS_DIR, exist_ok=True)
        nfiles = generate_legacy_corpus(CORPUS_DIR)
        print(f"Generated {nfiles} corpus files in {CORPUS_DIR}")

    if (args.write_questions or not args.preset) and not args.corpus_only:
        _write_questions_file()


if __name__ == "__main__":
    main()

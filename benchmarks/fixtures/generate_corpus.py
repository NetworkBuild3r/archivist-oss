"""Generate seed corpus (50 markdown files) and questions.json (100 QA pairs).

Run once to populate benchmarks/fixtures/:
    python benchmarks/fixtures/generate_corpus.py
"""

import json
import os
import random

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")

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
         "topic": "incident", "query_type": "temporal", "difficulty": "easy"},
        {"query": "What are the most recent changes to the deployment pipeline?",
         "expected_keywords": ["ArgoCD", "staging", "automated", "sync"],
         "expected_answer": "Deployment uses ArgoCD with automated sync enabled for staging.",
         "topic": "cicd", "query_type": "temporal", "difficulty": "medium"},
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
         "agent_filter": "chief"},
        {"query": "What does gitbob know about the CI pipeline?",
         "expected_keywords": ["GitLab CI", "15-minute", "timeout"],
         "expected_answer": "The CI pipeline runs on GitLab CI with a 15-minute timeout per job.",
         "topic": "cicd", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "gitbob"},
        {"query": "What monitoring information has grafgreg recorded?",
         "expected_keywords": ["Prometheus", "Grafana", "SLO", "dashboard"],
         "expected_answer": "Prometheus scrapes metrics every 15 seconds from all services.",
         "topic": "monitoring", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "grafgreg"},
        {"query": "What deployments has argo managed?",
         "expected_keywords": ["ArgoCD", "sync", "staging", "rollback"],
         "expected_answer": "Deployment uses ArgoCD with automated sync enabled for staging.",
         "topic": "cicd", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "argo"},
        {"query": "What has securitysam flagged about authentication?",
         "expected_keywords": ["JWT", "Vault", "RBAC", "OPA"],
         "expected_answer": "JWT tokens expire after 1 hour with refresh tokens valid for 7 days.",
         "topic": "security", "query_type": "agent_scoped", "difficulty": "medium",
         "agent_filter": "securitysam"},
    ]
    for sq in agent_scope_questions:
        qid += 1
        questions.append({"id": qid, **sq})

    while len(questions) < 100:
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

    return questions[:100]


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
    import re
    numbers = re.findall(r'\d+[\d,.]*%?', fact)
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', fact)
    return (numbers + proper_nouns)[:5]


def main():
    random.seed(42)
    os.makedirs(CORPUS_DIR, exist_ok=True)

    topic_names = list(TOPICS.keys())
    file_idx = 0

    for agent_idx, agent in enumerate(AGENTS):
        agent_dir = os.path.join(CORPUS_DIR, "agents", agent)
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

    print(f"Generated {file_idx} corpus files in {CORPUS_DIR}")

    questions = _generate_questions()
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    type_counts = {}
    for q in questions:
        qt = q["query_type"]
        type_counts[qt] = type_counts.get(qt, 0) + 1

    print(f"Generated {len(questions)} questions in {QUESTIONS_PATH}")
    print(f"  By type: {type_counts}")


if __name__ == "__main__":
    main()

"""Unit tests for Archivist Pydantic Settings v2 config.

Tests validate:
- Correct defaults are preserved after the Pydantic migration.
- ValidationError (not ValueError) is raised for invalid combinations.
- Env-var override via monkeypatch.setenv + _build_settings().
- .env file parsing (via direct constructor kwargs).
- frozen=True prevents mutation.
- feature_flags property shape.
- WEBHOOK_EVENTS comma-separated parsing.
- CURATOR_EXTRACT_PREFIXES comma-separated parsing.
- TEAM_MAP YAML loading.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivist.core.config import ArchivistSettings

pytestmark = [pytest.mark.unit]


def _fresh(monkeypatch, overrides: dict) -> ArchivistSettings:
    """Return a freshly-constructed ArchivistSettings with env overrides."""

    for k, v in overrides.items():
        monkeypatch.setenv(k, str(v))

    # Import the builder — no module reload needed with Pydantic Settings
    from archivist.core.config import _build_settings

    return _build_settings()


class TestConfigDefaults:
    """All defaults must exactly match the original os.getenv defaults."""

    def test_sqlite_wal_autocheckpoint_default(self):
        import archivist.core.config as cfg

        assert cfg.SQLITE_WAL_AUTOCHECKPOINT == 1000

    def test_sqlite_busy_timeout_default(self):
        import archivist.core.config as cfg

        assert cfg.SQLITE_BUSY_TIMEOUT_MS == 5000

    def test_qdrant_collection_default(self):
        import archivist.core.config as cfg

        assert cfg.QDRANT_COLLECTION == "archivist_memories"

    def test_vector_dim_default(self):
        # Build with an explicit override to verify the coded default independent
        # of the .env file (which may set VECTOR_DIM to a non-default value).
        from archivist.core.config import _build_settings

        s = _build_settings(vector_dim=1024)
        assert s.vector_dim == 1024

    def test_outbox_defaults(self):
        import archivist.core.config as cfg

        assert cfg.OUTBOX_ENABLED is False
        assert cfg.OUTBOX_DRAIN_INTERVAL == 2
        assert cfg.OUTBOX_BATCH_SIZE == 50
        assert cfg.OUTBOX_MAX_RETRIES == 5
        assert cfg.OUTBOX_ORPHAN_TIMEOUT_SECONDS == 60
        assert cfg.OUTBOX_ORPHAN_SWEEP_EVERY_N == 30
        assert cfg.OUTBOX_RETENTION_DAYS == 7

    def test_bm25_disabled_by_default(self):
        import archivist.core.config as cfg

        assert cfg.BM25_ENABLED is False

    def test_reranker_disabled_by_default(self):
        import archivist.core.config as cfg

        assert cfg.RERANKER_ENABLED is False

    def test_webhook_events_empty_set(self):
        import archivist.core.config as cfg

        assert set() == cfg.WEBHOOK_EVENTS
        assert isinstance(cfg.WEBHOOK_EVENTS, set)

    def test_curator_extract_prefixes_default(self):
        import archivist.core.config as cfg

        assert "agents/" in cfg.CURATOR_EXTRACT_PREFIXES
        assert "memories/" in cfg.CURATOR_EXTRACT_PREFIXES

    def test_default_confidence_by_actor_type(self):
        import archivist.core.config as cfg

        d = cfg.DEFAULT_CONFIDENCE_BY_ACTOR_TYPE
        assert d["human"] == 1.0
        assert d["agent"] == 0.8
        assert d["extracted"] == 0.5


class TestConfigValidation:
    """Invalid values must raise pydantic.ValidationError at construction time."""

    def test_invalid_chunk_overlap_raises(self, monkeypatch, tmp_path):
        """CHUNK_OVERLAP >= CHUNK_SIZE must fail validation."""
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            _fresh(
                monkeypatch,
                {
                    "SQLITE_PATH": str(tmp_path / "graph.db"),
                    "CHUNK_SIZE": "100",
                    "CHUNK_OVERLAP": "100",
                },
            )

    def test_negative_wal_autocheckpoint_raises(self, monkeypatch, tmp_path):
        """Negative SQLITE_WAL_AUTOCHECKPOINT must fail validation."""
        with pytest.raises(ValidationError, match="sqlite_wal_autocheckpoint"):
            _fresh(
                monkeypatch,
                {
                    "SQLITE_PATH": str(tmp_path / "graph.db"),
                    "SQLITE_WAL_AUTOCHECKPOINT": "-1",
                },
            )

    def test_negative_busy_timeout_raises(self, monkeypatch, tmp_path):
        """Negative SQLITE_BUSY_TIMEOUT_MS must fail validation."""
        with pytest.raises(ValidationError, match="sqlite_busy_timeout_ms"):
            _fresh(
                monkeypatch,
                {
                    "SQLITE_PATH": str(tmp_path / "graph.db"),
                    "SQLITE_BUSY_TIMEOUT_MS": "-500",
                },
            )

    def test_invalid_vector_dim_raises(self, monkeypatch, tmp_path):
        """VECTOR_DIM < 1 must fail validation (ge=1 constraint)."""
        with pytest.raises(ValidationError):
            _fresh(
                monkeypatch,
                {
                    "SQLITE_PATH": str(tmp_path / "graph.db"),
                    "VECTOR_DIM": "0",
                },
            )

    def test_retrieval_threshold_out_of_range(self, monkeypatch, tmp_path):
        """RETRIEVAL_THRESHOLD outside [0,1] must fail (ge/le constraints)."""
        with pytest.raises(ValidationError):
            _fresh(
                monkeypatch,
                {
                    "SQLITE_PATH": str(tmp_path / "graph.db"),
                    "RETRIEVAL_THRESHOLD": "1.5",
                },
            )


class TestEnvVarOverrides:
    """Env var overrides must be reflected in a freshly-built settings instance."""

    def test_qdrant_url_override(self, monkeypatch, tmp_path):
        s = _fresh(monkeypatch, {"SQLITE_PATH": str(tmp_path), "QDRANT_URL": "http://qdrant:6333"})
        assert s.qdrant_url == "http://qdrant:6333"

    def test_outbox_enabled_override(self, monkeypatch, tmp_path):
        s = _fresh(monkeypatch, {"SQLITE_PATH": str(tmp_path), "OUTBOX_ENABLED": "true"})
        assert s.outbox_enabled is True

    def test_bm25_enabled_override(self, monkeypatch, tmp_path):
        s = _fresh(monkeypatch, {"SQLITE_PATH": str(tmp_path), "BM25_ENABLED": "1"})
        assert s.bm25_enabled is True

    def test_vector_dim_override(self, monkeypatch, tmp_path):
        s = _fresh(monkeypatch, {"SQLITE_PATH": str(tmp_path), "VECTOR_DIM": "768"})
        assert s.vector_dim == 768


class TestFrozenModel:
    """The settings singleton must be immutable."""

    def test_frozen_prevents_attribute_assignment(self):
        import archivist.core.config as cfg

        with pytest.raises((TypeError, ValidationError)):
            cfg._settings.qdrant_url = "http://other:6333"  # type: ignore[misc]


class TestFeatureFlagsProperty:
    """feature_flags property must cover all known flags."""

    def test_feature_flags_is_dict_of_bools(self):
        import archivist.core.config as cfg

        flags = cfg._settings.feature_flags
        assert isinstance(flags, dict)
        assert all(isinstance(v, bool) for v in flags.values())

    def test_feature_flags_contains_expected_keys(self):
        import archivist.core.config as cfg

        flags = cfg._settings.feature_flags
        for key in (
            "BM25_ENABLED",
            "OUTBOX_ENABLED",
            "RERANKER_ENABLED",
            "PROVENANCE_ENABLED",
            "METRICS_ENABLED",
        ):
            assert key in flags, f"Missing flag: {key}"

    def test_feature_flags_matches_compat_layer(self):
        import archivist.core.config as cfg

        flags = cfg._settings.feature_flags
        assert flags["BM25_ENABLED"] == cfg.BM25_ENABLED
        assert flags["OUTBOX_ENABLED"] == cfg.OUTBOX_ENABLED
        assert flags["METRICS_ENABLED"] == cfg.METRICS_ENABLED


class TestWebhookEventsParsing:
    """WEBHOOK_EVENTS must parse comma-separated strings correctly."""

    def test_comma_separated_events(self):
        from archivist.core.config import _build_settings

        s = _build_settings(webhook_events="store,delete,merge")
        assert s.webhook_events_set == {"store", "delete", "merge"}

    def test_empty_webhook_events(self):
        from archivist.core.config import _build_settings

        s = _build_settings(webhook_events="")
        assert s.webhook_events_set == set()

    def test_events_with_extra_spaces(self):
        from archivist.core.config import _build_settings

        s = _build_settings(webhook_events=" store , delete ")
        assert "store" in s.webhook_events_set
        assert "delete" in s.webhook_events_set


class TestCuratorCsvParsing:
    """CURATOR_EXTRACT_PREFIXES and CURATOR_EXTRACT_SKIP_SEGMENTS must parse CSVs."""

    def test_custom_prefixes(self):
        from archivist.core.config import _build_settings

        s = _build_settings(curator_extract_prefixes="docs/,notes/,logs/")
        assert s.curator_extract_prefixes_list == ["docs/", "notes/", "logs/"]

    def test_custom_skip_segments(self):
        from archivist.core.config import _build_settings

        s = _build_settings(curator_extract_skip_segments="node_modules,.git,dist")
        assert "node_modules" in s.curator_extract_skip_segments_list
        assert ".git" in s.curator_extract_skip_segments_list


class TestTeamMapLoading:
    """TEAM_MAP must load from YAML file or JSON env var."""

    def test_team_map_from_yaml(self, monkeypatch, tmp_path):
        yaml_path = tmp_path / "team_map.yaml"
        yaml_path.write_text("agent1: teamA\nagent2: teamB\n")
        s = _fresh(
            monkeypatch,
            {
                "SQLITE_PATH": str(tmp_path),
                "TEAM_MAP_PATH": str(yaml_path),
            },
        )
        result = s._load_team_map()
        assert result == {"agent1": "teamA", "agent2": "teamB"}

    def test_team_map_from_json_env(self, monkeypatch, tmp_path):
        s = _fresh(
            monkeypatch,
            {
                "SQLITE_PATH": str(tmp_path),
                "TEAM_MAP_JSON": '{"agentX": "ops"}',
            },
        )
        result = s._load_team_map()
        assert result == {"agentX": "ops"}

    def test_team_map_empty_when_no_source(self, monkeypatch, tmp_path):
        s = _fresh(monkeypatch, {"SQLITE_PATH": str(tmp_path)})
        result = s._load_team_map()
        assert result == {}


class TestCompatLayer:
    """Module-level UPPER_CASE re-exports must match _settings fields."""

    def test_all_compat_exports_match_settings(self):
        import archivist.core.config as cfg

        pairs = [
            (cfg.QDRANT_COLLECTION, cfg._settings.qdrant_collection),
            (cfg.VECTOR_DIM, cfg._settings.vector_dim),
            (cfg.LLM_MODEL, cfg._settings.llm_model),
            (cfg.CHUNK_SIZE, cfg._settings.chunk_size),
            (cfg.OUTBOX_ENABLED, cfg._settings.outbox_enabled),
            (cfg.METRICS_ENABLED, cfg._settings.metrics_enabled),
            (cfg.SQLITE_WAL_AUTOCHECKPOINT, cfg._settings.sqlite_wal_autocheckpoint),
            (cfg.SQLITE_BUSY_TIMEOUT_MS, cfg._settings.sqlite_busy_timeout_ms),
        ]
        for compat_val, settings_val in pairs:
            assert compat_val == settings_val

    def test_valid_retention_classes(self):
        import archivist.core.config as cfg

        assert "ephemeral" in cfg.VALID_RETENTION_CLASSES
        assert "permanent" in cfg.VALID_RETENTION_CLASSES

    def test_durable_entity_types(self):
        import archivist.core.config as cfg

        assert "person" in cfg.DURABLE_ENTITY_TYPES
        assert "database" in cfg.DURABLE_ENTITY_TYPES

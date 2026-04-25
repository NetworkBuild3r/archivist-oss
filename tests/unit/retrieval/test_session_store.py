"""Unit tests for retrieval/session_store.py."""

from __future__ import annotations

import time

import pytest

from archivist.retrieval.session_store import SessionStore, get_session_store


class TestSessionStorePut:
    def test_put_and_get_roundtrip(self):
        s = SessionStore()
        s.put("alice", "s1", "goal", "fix the bug")
        assert s.get("alice", "s1", "goal") == "fix the bug"

    def test_overwrite_existing_key(self):
        s = SessionStore()
        s.put("alice", "s1", "key", "v1")
        s.put("alice", "s1", "key", "v2")
        assert s.get("alice", "s1", "key") == "v2"

    def test_different_sessions_are_isolated(self):
        s = SessionStore()
        s.put("alice", "s1", "note", "session 1")
        s.put("alice", "s2", "note", "session 2")
        assert s.get("alice", "s1", "note") == "session 1"
        assert s.get("alice", "s2", "note") == "session 2"

    def test_different_agents_are_isolated(self):
        s = SessionStore()
        s.put("alice", "s1", "k", "alice-val")
        s.put("bob", "s1", "k", "bob-val")
        assert s.get("alice", "s1", "k") == "alice-val"
        assert s.get("bob", "s1", "k") == "bob-val"

    def test_missing_key_returns_none(self):
        s = SessionStore()
        assert s.get("alice", "s1", "missing") is None


class TestSessionStoreTTL:
    def test_expired_entry_returns_none(self):
        s = SessionStore(default_ttl_seconds=1)
        s.put("alice", "s1", "k", "v", ttl_seconds=0)
        time.sleep(0.01)
        assert s.get("alice", "s1", "k") is None

    def test_non_expired_entry_is_returned(self):
        s = SessionStore(default_ttl_seconds=3600)
        s.put("alice", "s1", "k", "alive")
        assert s.get("alice", "s1", "k") == "alive"

    def test_custom_ttl_overrides_default(self):
        s = SessionStore(default_ttl_seconds=3600)
        s.put("alice", "s1", "k", "short", ttl_seconds=0)
        time.sleep(0.01)
        assert s.get("alice", "s1", "k") is None


class TestSessionStoreFlush:
    def test_flush_returns_all_live_entries(self):
        s = SessionStore()
        s.put("alice", "s1", "a", "alpha")
        s.put("alice", "s1", "b", "beta")
        entries = s.flush("alice", "s1")
        keys = {e["key"] for e in entries}
        assert keys == {"a", "b"}

    def test_flush_empties_session(self):
        s = SessionStore()
        s.put("alice", "s1", "k", "v")
        s.flush("alice", "s1")
        assert s.get("alice", "s1", "k") is None

    def test_flush_does_not_affect_other_sessions(self):
        s = SessionStore()
        s.put("alice", "s1", "k", "v1")
        s.put("alice", "s2", "k", "v2")
        s.flush("alice", "s1")
        assert s.get("alice", "s2", "k") == "v2"

    def test_flush_drops_expired_entries(self):
        s = SessionStore()
        s.put("alice", "s1", "live", "val", ttl_seconds=3600)
        s.put("alice", "s1", "dead", "val", ttl_seconds=0)
        time.sleep(0.01)
        entries = s.flush("alice", "s1")
        keys = {e["key"] for e in entries}
        assert "live" in keys
        assert "dead" not in keys

    def test_flush_empty_session_returns_empty_list(self):
        s = SessionStore()
        assert s.flush("nobody", "nosession") == []


class TestSessionStorePromote:
    def test_promote_marks_entry(self):
        s = SessionStore()
        s.put("alice", "s1", "goal", "ship the feature")
        text = s.promote("alice", "s1", "goal")
        assert text == "ship the feature"

    def test_promoted_flag_appears_in_flush(self):
        s = SessionStore()
        s.put("alice", "s1", "p", "promote me")
        s.promote("alice", "s1", "p")
        entries = s.flush("alice", "s1")
        assert entries[0]["promoted"] is True

    def test_promote_missing_key_returns_none(self):
        s = SessionStore()
        assert s.promote("alice", "s1", "nokey") is None


class TestSessionStoreCapacity:
    def test_evicts_oldest_when_full(self):
        s = SessionStore(max_entries=3)
        s.put("a", "s", "k1", "v1")
        s.put("a", "s", "k2", "v2")
        s.put("a", "s", "k3", "v3")
        s.put("a", "s", "k4", "v4")  # k1 should be evicted
        assert s.size() == 3
        assert s.get("a", "s", "k1") is None
        assert s.get("a", "s", "k4") == "v4"


class TestSessionStoreListKeys:
    def test_list_keys_returns_live_keys(self):
        s = SessionStore()
        s.put("alice", "s1", "x", "1")
        s.put("alice", "s1", "y", "2")
        keys = s.list_keys("alice", "s1")
        assert set(keys) == {"x", "y"}

    def test_list_keys_excludes_expired(self):
        s = SessionStore()
        s.put("alice", "s1", "live", "v", ttl_seconds=3600)
        s.put("alice", "s1", "dead", "v", ttl_seconds=0)
        time.sleep(0.01)
        keys = s.list_keys("alice", "s1")
        assert "dead" not in keys
        assert "live" in keys


class TestSessionStoreDelete:
    def test_delete_removes_entry(self):
        s = SessionStore()
        s.put("a", "s", "k", "v")
        assert s.delete("a", "s", "k") is True
        assert s.get("a", "s", "k") is None

    def test_delete_missing_key_returns_false(self):
        s = SessionStore()
        assert s.delete("a", "s", "missing") is False


class TestSessionStoreSingleton:
    def test_get_session_store_returns_same_instance(self, monkeypatch):
        import archivist.retrieval.session_store as ss_mod

        monkeypatch.setattr(ss_mod, "_store_instance", None)
        inst1 = get_session_store()
        inst2 = get_session_store()
        assert inst1 is inst2

    def test_singleton_is_session_store_type(self, monkeypatch):
        import archivist.retrieval.session_store as ss_mod

        monkeypatch.setattr(ss_mod, "_store_instance", None)
        inst = get_session_store()
        assert isinstance(inst, SessionStore)

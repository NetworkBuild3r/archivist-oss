/* INIT-013/SPEC-003 — observability billboard (XSS-safe textContent only) */
(function () {
  "use strict";

  var KEY_STORAGE = "archivist.admin_ui.api_key";
  var apiKey = "";

  function el(id) {
    return document.getElementById(id);
  }

  function text(node, value) {
    node.textContent = value == null ? "" : String(value);
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function cardRow(label, value) {
    var wrap = document.createElement("div");
    var k = document.createElement("span");
    k.className = "k";
    text(k, label);
    var v = document.createElement("span");
    v.className = "v";
    text(v, value);
    wrap.appendChild(k);
    wrap.appendChild(document.createTextNode(" "));
    wrap.appendChild(v);
    return wrap;
  }

  function setStatus(msg, kind) {
    var n = el("auth-status");
    n.className = "auth-status" + (kind ? " " + kind : "");
    text(n, msg);
  }

  function headers() {
    var h = { Accept: "application/json" };
    if (apiKey) h["X-API-Key"] = apiKey;
    return h;
  }

  function apiUrl(path, params) {
    var u = new URL(path, window.location.origin);
    if (params) {
      Object.keys(params).forEach(function (k) {
        var v = params[k];
        if (v != null && String(v).trim() !== "") u.searchParams.set(k, String(v).trim());
      });
    }
    return u.toString();
  }

  async function fetchJson(path, params) {
    var resp = await fetch(apiUrl(path, params), { headers: headers() });
    var body = null;
    try {
      body = await resp.json();
    } catch (_e) {
      body = null;
    }
    if (resp.status === 401) {
      setStatus("Unauthorized — set API key (X-API-Key / Bearer).", "err");
      throw new Error("unauthorized");
    }
    if (!resp.ok) {
      var reason =
        (body && (body.reason || body.error)) || ("HTTP " + resp.status);
      throw new Error(String(reason));
    }
    return body;
  }

  function kpi(label, value) {
    var d = document.createElement("div");
    d.className = "kpi";
    var l = document.createElement("div");
    l.className = "label";
    text(l, label);
    var v = document.createElement("div");
    v.className = "value";
    text(v, value);
    d.appendChild(l);
    d.appendChild(v);
    return d;
  }

  function fmtNum(n) {
    if (n == null || n === "") return "—";
    if (typeof n === "number") return Number.isInteger(n) ? String(n) : n.toFixed(2);
    return String(n);
  }

  async function loadHealth() {
    var row = el("kpi-row");
    clear(row);
    text(el("health-detail"), "");
    el("health-detail").hidden = true;
    try {
      var data = await fetchJson("/admin/dashboard", { window_days: "7" });
      setStatus("Connected.", "ok");
      var ts = data.token_savings || {};
      row.appendChild(kpi("Tokens saved", fmtNum(ts.total_tokens_saved)));
      row.appendChild(kpi("Savings %", fmtNum(ts.avg_savings_pct)));
      row.appendChild(kpi("Returned", fmtNum(ts.total_tokens_returned)));
      row.appendChild(kpi("Naive", fmtNum(ts.total_tokens_naive)));
      if (ts.estimated_usd_saved != null) {
        row.appendChild(kpi("USD saved (est.)", fmtNum(ts.estimated_usd_saved)));
      }
      var health = data.health || data.memory_health || {};
      if (health && typeof health === "object") {
        Object.keys(health).slice(0, 4).forEach(function (k) {
          var v = health[k];
          if (v != null && typeof v !== "object") row.appendChild(kpi(k, fmtNum(v)));
        });
      }
      el("health-detail").hidden = false;
      text(el("health-detail"), JSON.stringify(data, null, 2));
    } catch (e) {
      row.appendChild(kpi("Error", e.message || String(e)));
    }
  }

  async function loadLineage(ev) {
    if (ev) ev.preventDefault();
    var form = el("lineage-form");
    var fd = new FormData(form);
    var params = {
      memory_id: fd.get("memory_id"),
      entity_id: fd.get("entity_id"),
      namespace: fd.get("namespace"),
      agent_id: fd.get("agent_id"),
      limit: fd.get("limit"),
    };
    var meta = el("lineage-meta");
    var list = el("lineage-edges");
    clear(list);
    text(meta, "Loading…");
    try {
      var data = await fetchJson("/admin/lineage", params);
      text(
        meta,
        (data.resource_type || "resource") +
          " " +
          (data.resource_id || "") +
          " · ns " +
          (data.namespace || "—") +
          " · " +
          (data.edge_count != null ? data.edge_count : (data.edges || []).length) +
          " edges"
      );
      var edges = data.edges || [];
      if (!edges.length) {
        var empty = document.createElement("p");
        empty.className = "empty";
        text(empty, "No edges.");
        list.appendChild(empty);
        return;
      }
      edges.forEach(function (edge) {
        var c = document.createElement("div");
        c.className = "card";
        var row = document.createElement("div");
        row.className = "row";
        row.appendChild(cardRow("type", edge.edge_type || edge.type || "—"));
        row.appendChild(cardRow("from", edge.from_id || edge.source || "—"));
        row.appendChild(cardRow("to", edge.to_id || edge.target || "—"));
        if (edge.relation) row.appendChild(cardRow("relation", edge.relation));
        c.appendChild(row);
        list.appendChild(c);
      });
    } catch (e) {
      text(meta, e.message || String(e));
    }
  }

  async function loadAudit(ev) {
    if (ev) ev.preventDefault();
    var form = el("audit-form");
    var fd = new FormData(form);
    var params = {
      memory_id: fd.get("memory_id"),
      agent_id: fd.get("agent_id"),
      limit: fd.get("limit"),
    };
    var meta = el("audit-meta");
    var list = el("audit-entries");
    clear(list);
    text(meta, "Loading…");
    try {
      var data = await fetchJson("/admin/audit", params);
      var entries = data.entries || [];
      text(meta, (data.count != null ? data.count : entries.length) + " entries");
      if (!entries.length) {
        var empty = document.createElement("p");
        empty.className = "empty";
        text(empty, "No audit entries.");
        list.appendChild(empty);
        return;
      }
      entries.forEach(function (entry) {
        var c = document.createElement("div");
        c.className = "card";
        var row = document.createElement("div");
        row.className = "row";
        Object.keys(entry).forEach(function (k) {
          var v = entry[k];
          if (v != null && typeof v !== "object") row.appendChild(cardRow(k, v));
        });
        c.appendChild(row);
        list.appendChild(c);
      });
    } catch (e) {
      text(meta, e.message || String(e));
    }
  }

  async function loadRetrieval(ev) {
    if (ev) ev.preventDefault();
    var form = el("retrieval-form");
    var fd = new FormData(form);
    var params = {
      agent_id: fd.get("agent_id"),
      limit: fd.get("limit"),
    };
    var meta = el("retrieval-meta");
    var list = el("retrieval-rows");
    clear(list);
    text(meta, "Loading…");
    try {
      var data = await fetchJson("/admin/retrieval-logs", params);
      var logs = Array.isArray(data) ? data : data.logs || data.entries || [];
      text(meta, logs.length + " logs");
      if (!logs.length) {
        var empty = document.createElement("p");
        empty.className = "empty";
        text(empty, "No retrieval logs.");
        list.appendChild(empty);
        return;
      }
      logs.forEach(function (log) {
        var c = document.createElement("div");
        c.className = "card";
        var row = document.createElement("div");
        row.className = "row";
        ["agent_id", "tokens_returned", "tokens_naive", "savings_pct", "pack_policy", "created_at"].forEach(
          function (k) {
            if (log[k] != null) row.appendChild(cardRow(k, log[k]));
          }
        );
        c.appendChild(row);
        list.appendChild(c);
      });
    } catch (e) {
      text(meta, e.message || String(e));
    }
  }

  function showPanel(name) {
    ["health", "lineage", "audit", "retrieval"].forEach(function (p) {
      var panel = el("panel-" + p);
      var on = p === name;
      panel.hidden = !on;
      panel.classList.toggle("active", on);
    });
    document.querySelectorAll(".tabs button").forEach(function (btn) {
      btn.setAttribute("aria-selected", btn.getAttribute("data-panel") === name ? "true" : "false");
    });
  }

  function initAuth() {
    var remembered = sessionStorage.getItem(KEY_STORAGE) || "";
    if (remembered) {
      apiKey = remembered;
      el("api-key").value = remembered;
      el("remember-key").checked = true;
      setStatus("Key restored from sessionStorage.", "ok");
    }
    el("auth-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      apiKey = (el("api-key").value || "").trim();
      if (el("remember-key").checked && apiKey) {
        sessionStorage.setItem(KEY_STORAGE, apiKey);
      } else {
        sessionStorage.removeItem(KEY_STORAGE);
      }
      setStatus(apiKey ? "Key applied for this session." : "No key — open-mode server only.", "ok");
      loadHealth();
    });
  }

  function initTabs() {
    document.querySelectorAll(".tabs button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        showPanel(btn.getAttribute("data-panel"));
      });
    });
  }

  el("btn-refresh-health").addEventListener("click", loadHealth);
  el("btn-refresh-retrieval").addEventListener("click", loadRetrieval);
  el("lineage-form").addEventListener("submit", loadLineage);
  el("audit-form").addEventListener("submit", loadAudit);
  el("retrieval-form").addEventListener("submit", loadRetrieval);

  initAuth();
  initTabs();
  showPanel("health");
  loadHealth();
})();

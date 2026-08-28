// app.js
// ======
// Thin client for AgentHub Studio's API (see app.py). Owns UI state
// (thread id, chat history, current view/mode/db) and DOM rendering; all
// product logic (routing, SQL generation/validation, schema, MCP) lives
// server-side. Recent-chat titles/transcripts are cached in localStorage
// only, purely for this browser's convenience - never sent anywhere.

const jsonHeaders = { "Content-Type": "application/json" };

async function apiJson(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed.");
  return data;
}

const API = {
  bootstrap: () => fetch("/api/bootstrap").then(r => r.json()),
  databases: () => fetch("/api/databases").then(r => r.json()),
  databasesDetail: () => fetch("/api/databases/detail").then(apiJson),
  deleteDatabase: (dbName) => fetch("/api/databases/delete", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ db_name: dbName }),
  }).then(apiJson),
  ragDocuments: () => fetch("/api/rag/documents").then(apiJson),
  uploadRagDocument: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/rag/upload", { method: "POST", body: form }).then(apiJson);
  },
  queryRag: (body) => fetch("/api/rag/query", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  }).then(apiJson),
  deleteRagDocument: (documentId) => fetch(`/api/rag/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  }).then(apiJson),
  schema: (dbName) => fetch(`/api/schema?db_name=${encodeURIComponent(dbName || "")}`).then(r => r.json()),
  settings: () => fetch("/api/settings").then(r => r.json()),
  mcpTools: () => fetch("/api/mcp-tools").then(r => r.json()),
  integrationsStatus: () => fetch("/api/integrations/status").then(r => r.json()),
  disconnectIntegration: (provider) => fetch("/api/integrations/disconnect", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ provider }),
  }).then(r => r.json()),
  chat: (body) => fetch("/api/chat", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }).then(r => r.json()),
  approve: (body) => fetch("/api/approve", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }).then(r => r.json()),
  cancel: (body) => fetch("/api/cancel", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }).then(r => r.json()),
  newChat: (body) => fetch("/api/new-chat", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }).then(r => r.json()),
  mcpRun: (body) => fetch("/api/mcp-run", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }).then(r => r.json()),
};

const RECENT_CHATS_KEY = "agenthub_recent_chats";
const THEME_KEY = "agenthub_theme";

const state = {
  threadId: null,
  view: "chat",
  mode: "Auto",
  dbName: null,
  includeSampleData: true,
  chatHistory: [], // {role, content, timestamp, meta}
  lastQuestion: "",
  currentSchema: { db_name: null, tables: [] },
  selectedSchemaTable: null,
  lastResult: null,
  currentSql: "",
  buildExamples: [],
  queryExamples: [],
  unavailableModes: [],
  pendingDeleteDbName: null,
  pendingDeleteDocumentId: null,
  ragDocuments: [],
  // Conversation-scoped attachments: ready document ids currently in chat scope.
  attachedDocumentIds: [],
  // Local upload/index chips: { localId, file_name, status, document_id?, error? }
  uploadChips: [],
  toastTimer: null,
};

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function preferredTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch (_) {
    /* Storage can be unavailable in privacy-restricted contexts. */
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme, persist = true) {
  const nextTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = nextTheme;

  const toggle = el("theme-switch");
  if (toggle) {
    const isDark = nextTheme === "dark";
    toggle.setAttribute("aria-checked", String(isDark));
    toggle.setAttribute("aria-label", isDark ? "Use light mode" : "Use dark mode");
  }

  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, nextTheme);
    } catch (_) {
      /* The theme still applies for this page when storage is unavailable. */
    }
  }
  window.dispatchEvent(new CustomEvent("agenthub:themechange", { detail: { theme: nextTheme } }));
}

function initTheme() {
  const initial = document.documentElement.dataset.theme || preferredTheme();
  applyTheme(initial, false);
}

function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function timeLabel(iso) {
  const d = iso ? new Date(iso) : new Date();
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message, kind = "success") {
  const toast = el("toast");
  toast.textContent = message;
  toast.className = `toast show ${kind === "error" ? "error" : "success"}`;
  if (state.toastTimer) window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.className = "toast";
  }, 3200);
}

function prettifyColumn(name) {
  return String(name).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const MONEY_HINTS = ["salary", "price", "cost", "amount", "revenue", "pay", "bonus", "total", "fee"];
function formatValue(colName, value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") {
    const isMoney = MONEY_HINTS.some((hint) => String(colName).toLowerCase().includes(hint));
    const rounded = Number.isInteger(value) ? value : Math.round(value * 100) / 100;
    const formatted = rounded.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return isMoney ? `$${formatted}` : formatted;
  }
  return String(value);
}

function downloadBlob(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Minimal markdown-ish rendering for chat bubbles
// ---------------------------------------------------------------------------
function renderMarkdownLite(text) {
  const escaped = escapeHtml(text);
  const lines = escaped.split("\n");
  const htmlLines = [];
  let inList = false;

  for (const line of lines) {
    const bulletMatch = line.match(/^\s*-\s+(.*)$/);
    const headerMatch = line.match(/^(#{1,4})\s+(.*)$/);

    if (bulletMatch) {
      if (!inList) { htmlLines.push("<ul>"); inList = true; }
      htmlLines.push(`<li>${inlineFormat(bulletMatch[1])}</li>`);
      continue;
    }
    if (inList) { htmlLines.push("</ul>"); inList = false; }

    if (headerMatch) {
      htmlLines.push(`<strong class="md-header">${inlineFormat(headerMatch[2])}</strong>`);
      continue;
    }
    if (line.trim() === "") { htmlLines.push("<br>"); continue; }
    htmlLines.push(`<div>${inlineFormat(line)}</div>`);
  }
  if (inList) htmlLines.push("</ul>");
  return htmlLines.join("");
}

function inlineFormat(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+?)`/g, "<code>$1</code>");
}

// ---------------------------------------------------------------------------
// SQL syntax highlighting (client-side, dependency-free)
// ---------------------------------------------------------------------------
const SQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON",
  "GROUP", "ORDER", "BY", "LIMIT", "OFFSET", "AS", "AND", "OR", "NOT", "NULL", "IS",
  "IN", "LIKE", "BETWEEN", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "WITH",
  "HAVING", "ASC", "DESC", "UNION", "ALL", "EXISTS",
];
const SQL_TOKEN_RE = new RegExp(
  `(?<comment>--[^\\n]*)|(?<string>'[^']*'|"[^"]*")|(?<number>\\b\\d+(?:\\.\\d+)?\\b)|` +
  `(?<func>\\b[A-Za-z_][A-Za-z0-9_]*\\b(?=\\s*\\())|(?<kw>\\b(?:${SQL_KEYWORDS.join("|")})\\b)`,
  "gi",
);

function highlightSqlLine(line) {
  let out = "";
  let lastIndex = 0;
  for (const m of line.matchAll(SQL_TOKEN_RE)) {
    out += escapeHtml(line.slice(lastIndex, m.index));
    const g = m.groups;
    let cls = "";
    if (g.comment) cls = "tok-comment";
    else if (g.string) cls = "tok-str";
    else if (g.number) cls = "tok-num";
    else if (g.func) cls = "tok-fn";
    else if (g.kw) cls = "tok-kw";
    out += `<span class="${cls}">${escapeHtml(m[0])}</span>`;
    lastIndex = m.index + m[0].length;
  }
  out += escapeHtml(line.slice(lastIndex));
  return out || "&nbsp;";
}

function renderSqlBox(sql) {
  const lines = sql.split("\n");
  const html = lines
    .map((line, i) => `<div class="sql-line"><span class="ln">${i + 1}</span><span class="code-content">${highlightSqlLine(line)}</span></div>`)
    .join("");
  el("sql-box").innerHTML = `<code>${html}</code>`;
}

// ---------------------------------------------------------------------------
// View routing
// ---------------------------------------------------------------------------
function switchView(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  el(`view-${view}`).classList.add("active");
  closeAllMenus();
  updateNavActiveStates();

  if (view === "databases") loadDatabasesView();
  else if (view === "tools") loadToolsView();
  else if (view === "mcp") loadMcpView();
  else if (view === "settings") loadSettingsView();
}

function updateNavActiveStates() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    const isChatQuickMode = btn.dataset.quickMode;
    if (state.view === "chat" && isChatQuickMode) {
      btn.classList.toggle("active", state.mode === isChatQuickMode);
    } else if (!isChatQuickMode) {
      btn.classList.toggle("active", state.view === btn.dataset.view);
    } else {
      btn.classList.remove("active");
    }
  });
}

function closeAllMenus() {
  el("options-menu").classList.add("hidden");
  el("tools-menu").classList.add("hidden");
  const attachMenu = el("attach-menu");
  if (attachMenu) attachMenu.classList.add("hidden");
  const attachBtn = el("attach-btn");
  if (attachBtn) attachBtn.setAttribute("aria-expanded", "false");
}

// ---------------------------------------------------------------------------
// Chat rendering
// ---------------------------------------------------------------------------
function buildSummaryCard(result) {
  if (!result || !result.columns || result.columns.length === 0 || result.rows.length !== 1) return "";
  const row = result.rows[0];
  const mainCol = result.columns[0];
  const secondCol = result.columns.length > 1 ? result.columns[1] : null;

  let html = `<div class="summary-card">
    <div class="summary-icon"><svg class="icon" viewBox="0 0 24 24"><path d="M4 20V10M12 20V4M20 20v-7"/></svg></div>
    <div class="summary-main">
      <div class="summary-label">${escapeHtml(prettifyColumn(mainCol))}</div>
      <div class="summary-value" title="${escapeHtml(formatValue(mainCol, row[0]))}">${escapeHtml(formatValue(mainCol, row[0]))}</div>
    </div>`;
  if (secondCol !== null) {
    html += `<div class="summary-secondary">
      <div class="summary-label">${escapeHtml(prettifyColumn(secondCol))}</div>
      <div class="summary-value">${escapeHtml(formatValue(secondCol, row[1]))}</div>
    </div>`;
  }
  html += `</div>`;
  return html;
}

function buildRowsNote(result) {
  if (!result || !result.rows || result.rows.length <= 1) return "";
  return `<div class="rows-note">
    <svg class="icon icon-sm" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/></svg>
    ${result.rows.length} rows returned &nbsp;·&nbsp; <button class="link-btn" data-action="view-results">View in Results tab</button>
  </div>`;
}

function buildBadges(meta) {
  if (!meta) return "";
  let badges = "<span class=\"msg-badges\">";
  if (meta.mode_used) {
    badges += `<span class="badge badge-muted">${escapeHtml(meta.mode_used)}</span>`;
  }
  if (!meta.sql) {
    badges += "</span>";
    return badges;
  }
  badges += `<span class="badge badge-sql">SQL</span>`;
  if (meta.sql_valid === true) {
    const title = meta.retry_count > 0 ? `Validated after ${meta.retry_count} repair${meta.retry_count > 1 ? "s" : ""}` : "Validated";
    badges += `<span class="badge-icon badge-icon-valid" title="${escapeHtml(title)}"><svg class="icon" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg></span>`;
  } else if (meta.sql_valid === false) {
    badges += `<span class="badge-icon badge-icon-invalid" title="Query failed"><svg class="icon" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg></span>`;
  }
  badges += `</span>`;
  return badges;
}

function renderMessage(msg, index) {
  const row = document.createElement("div");
  row.className = `msg ${msg.role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${msg.role === "user" ? "avatar-user" : "avatar-assistant"}`;
  avatar.textContent = msg.role === "user" ? "JD" : "AI";

  const body = document.createElement("div");
  body.className = "msg-body";

  const head = document.createElement("div");
  head.className = "msg-head";
  head.innerHTML = `
    <span class="msg-name">${msg.role === "user" ? "You" : "AgentHub Studio"}</span>
    ${buildBadges(msg.meta)}
    <span class="msg-time">${escapeHtml(timeLabel(msg.timestamp))}</span>
  `;

  const content = document.createElement("div");
  content.className = "msg-content";
  content.innerHTML = renderMarkdownLite(msg.content);
  if (msg.meta && msg.meta.result) {
    content.innerHTML += buildSummaryCard(msg.meta.result);
    content.innerHTML += buildRowsNote(msg.meta.result);
  }

  body.appendChild(head);
  body.appendChild(content);

  if (msg.role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    actions.innerHTML = `
      <button data-action="copy" title="Copy"><svg class="icon" viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
      <button data-action="up" title="Good response"><svg class="icon" viewBox="0 0 24 24"><path d="M7 10v12M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.34 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a2.5 2.5 0 0 1 3 2.5v1.38Z"/></svg></button>
      <button data-action="down" title="Needs work"><svg class="icon" viewBox="0 0 24 24"><path d="M17 14V2M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.34-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L13 22a2.5 2.5 0 0 1-3-2.5v-1.38Z"/></svg></button>
    `;
    if (msg.feedback === "up") actions.querySelector('[data-action="up"]').classList.add("active-up");
    if (msg.feedback === "down") actions.querySelector('[data-action="down"]').classList.add("active-down");

    actions.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const action = btn.dataset.action;
      if (action === "copy") {
        navigator.clipboard.writeText(msg.content);
      } else if (action === "up" || action === "down") {
        msg.feedback = msg.feedback === action ? null : action;
        actions.querySelector('[data-action="up"]').classList.toggle("active-up", msg.feedback === "up");
        actions.querySelector('[data-action="down"]').classList.toggle("active-down", msg.feedback === "down");
        persistRecentChat();
      }
    });
    body.appendChild(actions);
  }

  content.addEventListener("click", (e) => {
    if (e.target.dataset.action === "view-results") {
      activateTab("results");
    }
  });

  row.appendChild(avatar);
  row.appendChild(body);
  return row;
}

function renderChatMessages() {
  const box = el("chatbot");
  box.innerHTML = "";
  state.chatHistory.forEach((msg, i) => box.appendChild(renderMessage(msg, i)));
  box.scrollTop = box.scrollHeight;
}

function appendMessage(msg) {
  state.chatHistory.push(msg);
  renderComposerExamples();
  const box = el("chatbot");
  box.appendChild(renderMessage(msg, state.chatHistory.length - 1));
  box.scrollTop = box.scrollHeight;
  persistRecentChat();
}

// ---------------------------------------------------------------------------
// Inspector (Visualize / SQL / Results / Schema scroll together as one panel,
// like the query console this is modeled on; Debug / Sources are separate)
// ---------------------------------------------------------------------------
const COMBINED_TABS = ["visualize", "sql", "results", "schema"];

function setActiveTabButton(tab) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
}

function activateTab(tab) {
  setActiveTabButton(tab);
  if (COMBINED_TABS.includes(tab)) {
    el("combined-panel").classList.add("active");
    el("tab-debug").classList.remove("active");
    el("tab-sources").classList.remove("active");
    const section = el(`sec-${tab}`);
    if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
    if (tab === "visualize" && window.AgentHubVisualizer) {
      window.AgentHubVisualizer.onTabShow();
    }
  } else {
    el("combined-panel").classList.remove("active");
    el("tab-debug").classList.toggle("active", tab === "debug");
    el("tab-sources").classList.toggle("active", tab === "sources");
  }
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  // Scroll-spy: while the combined Visualize/SQL/Results/Schema panel is showing,
  // highlight whichever section the user has scrolled to.
  const observer = new IntersectionObserver(
    (entries) => {
      if (!el("combined-panel").classList.contains("active")) return;
      const visible = entries.filter((e) => e.isIntersecting);
      if (visible.length === 0) return;
      visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      setActiveTabButton(visible[0].target.id.replace("sec-", ""));
    },
    { root: el("inspector-body"), threshold: [0, 0.1, 0.5] },
  );
  COMBINED_TABS.forEach((tab) => {
    const section = el(`sec-${tab}`);
    if (section) observer.observe(section);
  });
}

function responseUsedSql(modeUsed, data) {
  const mode = modeUsed || "";
  if (/\bQuery Database\b/.test(mode) || /->\s*SQL\b/.test(mode)) return true;
  return Boolean(data && data.sql);
}

function clearActiveQueryContext(note) {
  state.currentSql = "";
  state.lastResult = null;
  renderSqlBox(`-- ${note}`);
  el("sql-badge").className = "badge badge-muted";
  el("sql-badge").textContent = "Not used";
  el("result-box").innerHTML = `<p class="empty-state">${escapeHtml(note)}</p>`;
  el("results-title").textContent = "Results";
  if (window.AgentHubVisualizer) window.AgentHubVisualizer.reset();
  el("status-dot").className = "status-dot";
  el("status-text").textContent = note;
}

function updateInspector(data) {
  const usedSql = responseUsedSql(data.mode_used, data);

  if (usedSql && data.sql !== null && data.sql !== undefined) {
    state.currentSql = data.sql;
    renderSqlBox(data.sql || "-- Empty query");
    const badge = el("sql-badge");
    if (data.sql_valid === true) {
      badge.className = "badge badge-valid";
      badge.textContent = data.retry_count > 0 ? `Validated (${data.retry_count} repair${data.retry_count > 1 ? "s" : ""})` : "Validated";
    } else if (data.sql_valid === false) {
      badge.className = "badge badge-invalid";
      badge.textContent = "Failed";
    } else {
      badge.className = "badge badge-muted";
      badge.textContent = "Not run";
    }
  } else if (data.mode_used && !usedSql) {
    // Keep history in chat; do not let a General/RAG/MCP turn inherit the prior SQL result.
    clearActiveQueryContext("This response did not use the database.");
  }

  if (usedSql && data.result) {
    state.lastResult = data.result;
    renderResultTable(data.result);
    el("results-title").textContent = `Results (${data.result.rows.length} row${data.result.rows.length === 1 ? "" : "s"})`;
    if (window.AgentHubVisualizer) window.AgentHubVisualizer.onResultChange(state.lastResult);
  }

  if (data.schema !== null && data.schema !== undefined) {
    state.currentSchema = data.schema;
    state.selectedSchemaTable = null;
    renderSchemaTab();
  }

  if (data.debug !== null && data.debug !== undefined) {
    el("debug-box").textContent = data.debug;
  }

  if (data.sources !== null && data.sources !== undefined) {
    renderSourcesTab(data.sources, data.mode_used);
  }

  if (data.db_choices) {
    const preferred = (data.db_value === undefined) ? state.dbName : data.db_value;
    populateDbSelect(data.db_choices, preferred || null);
  }

  if (usedSql && data.execution_ms !== null && data.execution_ms !== undefined) {
    const dot = el("status-dot");
    dot.className = `status-dot ${data.sql_valid ? "ok" : "err"}`;
    const rc = data.row_count ?? 0;
    el("status-text").textContent =
      `Query executed in ${data.execution_ms} ms · ${rc} row${rc === 1 ? "" : "s"} · ${timeLabel(data.timestamp)}`;
  }

  setApproveVisible(!!data.approve_visible);
}

function renderResultTable(result) {
  const box = el("result-box");
  if (!result.columns || result.columns.length === 0) {
    box.innerHTML = "<p class='empty-state'>No rows returned.</p>";
    return;
  }
  let html = "<table><thead><tr>";
  for (const col of result.columns) html += `<th>${escapeHtml(col)}</th>`;
  html += "</tr></thead><tbody>";
  for (const row of result.rows) {
    html += "<tr>";
    row.forEach((cell, i) => { html += `<td>${escapeHtml(cell === null ? "NULL" : String(cell))}</td>`; });
    html += "</tr>";
  }
  html += "</tbody></table>";
  html += `<div class="result-footer">${result.rows.length} row${result.rows.length === 1 ? "" : "s"} returned</div>`;
  box.innerHTML = html;
}

function renderSchemaTab() {
  const select = el("schema-table-select");
  const tables = state.currentSchema.tables || [];

  if (tables.length === 0) {
    select.innerHTML = "";
    select.disabled = true;
    el("schema-box").innerHTML = "<p class='empty-state'>No database selected.</p>";
    return;
  }
  select.disabled = false;

  const names = tables.map((t) => t.name);
  if (!state.selectedSchemaTable || !names.includes(state.selectedSchemaTable)) {
    state.selectedSchemaTable = names[0];
  }
  select.innerHTML = names.map((n) => `<option value="${escapeHtml(n)}" ${n === state.selectedSchemaTable ? "selected" : ""}>${escapeHtml(n)}</option>`).join("");

  const table = tables.find((t) => t.name === state.selectedSchemaTable);
  const fkByColumn = {};
  for (const fk of table.foreign_keys || []) fkByColumn[fk.column] = fk;

  let html = `<div class="schema-table-meta">${table.row_count ?? 0} row${table.row_count === 1 ? "" : "s"} · ${table.columns.length} column${table.columns.length === 1 ? "" : "s"}</div>`;
  html += "<table><thead><tr><th>Column</th><th>Type</th><th>Nullable</th><th>Key</th></tr></thead><tbody>";
  for (const col of table.columns) {
    let key = "&mdash;";
    if (col.pk) key = "<span class='key-pk'>PK</span>";
    else if (fkByColumn[col.name]) {
      const fk = fkByColumn[col.name];
      key = `<span class='key-fk'>FK &rarr; ${escapeHtml(fk.ref_table)}.${escapeHtml(fk.ref_column)}</span>`;
    }
    html += `<tr><td><code>${escapeHtml(col.name)}</code></td><td>${escapeHtml(col.type)}</td><td>${col.pk ? "NO" : (col.notnull ? "NO" : "YES")}</td><td>${key}</td></tr>`;
  }
  html += "</tbody></table>";
  el("schema-box").innerHTML = html;
}

function renderSourcesTab(sources, modeUsed) {
  const box = el("sources-box");
  if (!sources || sources.length === 0) {
    const note = (modeUsed || "").includes("RAG")
      ? "No document chunks were cited in this answer."
      : (modeUsed || "").includes("Query Database")
      ? "No tables from the selected database were referenced in this answer."
      : "This response didn't query a database or document source.";
    box.innerHTML = `<p class="empty-state">${escapeHtml(note)}</p>`;
    return;
  }
  box.innerHTML = sources.map((s, idx) => {
    const isDoc = !!s.file_name;
    const scorePart = (typeof s.score === "number")
      ? ` · Relevance: ${s.score.toFixed(2)}`
      : (s.score !== undefined && s.score !== null && s.score !== "" ? ` · Relevance: ${escapeHtml(s.score)}` : "");
    const loc = isDoc
      ? `${s.page != null ? `Page ${escapeHtml(s.page)}` : `Chunk ${escapeHtml(s.chunk_index ?? "")}`}${scorePart}`
      : `${escapeHtml(s.db_name)} · ${s.row_count} row${s.row_count === 1 ? "" : "s"}`;
    const detail = isDoc
      ? [
          `File: ${s.file_name}`,
          s.page != null ? `Page: ${s.page}` : null,
          s.chunk_index != null ? `Chunk: ${s.chunk_index}` : null,
          typeof s.score === "number" ? `Relevance: ${s.score.toFixed(4)}` : null,
          "",
          s.preview || "(no preview)",
        ].filter((line) => line !== null).join("\n")
      : "";
    return `
    <div class="source-card" data-source-idx="${idx}" ${isDoc ? 'tabindex="0" role="button"' : ""}>
      ${isDoc
        ? `<svg class="icon" viewBox="0 0 24 24"><path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8Z"/><path d="M14 3v5h5"/></svg>`
        : `<svg class="icon" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>`}
      <div>
        <div class="source-name">${escapeHtml(s.file_name || s.table || "Source")}</div>
        <div class="source-meta">${loc}</div>
        ${s.preview ? `<div class="source-preview">"${escapeHtml(s.preview)}"</div>` : ""}
        ${isDoc ? `<div class="source-detail hidden">${escapeHtml(detail)}</div>` : ""}
      </div>
    </div>`;
  }).join("");

  box.querySelectorAll(".source-card[data-source-idx]").forEach((card) => {
    if (!card.querySelector(".source-detail")) return;
    card.addEventListener("click", () => {
      const detail = card.querySelector(".source-detail");
      const open = detail.classList.toggle("hidden") === false;
      card.classList.toggle("is-expanded", open);
    });
  });
}

function setApproveVisible(visible) {
  el("approve-row").classList.toggle("hidden", !visible);
}

// ---------------------------------------------------------------------------
// Sidebar / header controls
// ---------------------------------------------------------------------------
function populateModeSelect(modes, unavailable) {
  const select = el("mode-select");
  select.innerHTML = modes.map((m) => {
    const disabled = unavailable.includes(m) ? "disabled" : "";
    const label = unavailable.includes(m) ? `${m} (soon)` : m;
    return `<option value="${escapeHtml(m)}" ${disabled}>${escapeHtml(label)}</option>`;
  }).join("");
  select.value = state.mode;
}

function populateDbSelect(choices, value) {
  const select = el("db-select");
  const options = [
    `<option value="">No database selected</option>`,
    ...choices.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`),
  ];
  select.innerHTML = options.join("");
  if (value && choices.includes(value)) select.value = value;
  else select.value = "";
  state.dbName = select.value || null;
}

function renderComposerExamples() {
  const container = el("composer-examples");
  if (state.chatHistory.length > 0) {
    container.innerHTML = "";
    return;
  }

  let examples = [];
  if (state.mode === "Build Database") examples = state.buildExamples;
  else if (state.mode === "Query Database") examples = state.queryExamples;
  else if (state.mode === "Auto") examples = [...state.buildExamples.slice(0, 1), ...state.queryExamples.slice(0, 2)];

  container.innerHTML = "";
  for (const ex of examples) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = ex;
    chip.addEventListener("click", () => {
      if (el("send-btn").disabled) return;
      el("message-box").value = "";
      autoResizeMessageBox();
      sendMessage(ex);
    });
    container.appendChild(chip);
  }
}

// ---------------------------------------------------------------------------
// Recent chats (localStorage only)
// ---------------------------------------------------------------------------
function loadAllRecentChats() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_CHATS_KEY) || "{}");
  } catch {
    return {};
  }
}

function persistRecentChat() {
  if (state.chatHistory.length === 0 && state.attachedDocumentIds.length === 0) return;
  const all = loadAllRecentChats();
  const firstUserMsg = state.chatHistory.find((m) => m.role === "user");
  const title = firstUserMsg
    ? firstUserMsg.content.slice(0, 48)
    : (state.attachedDocumentIds.length ? "Document chat" : "New conversation");
  all[state.threadId] = {
    threadId: state.threadId,
    title,
    mode: state.mode,
    dbName: state.dbName,
    attachedDocumentIds: [...state.attachedDocumentIds],
    messages: state.chatHistory,
    updatedAt: Date.now(),
  };
  localStorage.setItem(RECENT_CHATS_KEY, JSON.stringify(all));
  renderRecentChats();
}

function renderRecentChats() {
  const all = loadAllRecentChats();
  const list = Object.values(all).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 20);
  const container = el("recent-chats-list");

  if (list.length === 0) {
    container.innerHTML = "<div class='recent-chats-empty'>No conversations yet</div>";
    return;
  }

  container.innerHTML = "";
  for (const chat of list) {
    const item = document.createElement("button");
    item.className = `recent-chat-item ${chat.threadId === state.threadId ? "active" : ""}`;
    item.innerHTML = `
      <svg class="icon" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/></svg>
      <span class="title">${escapeHtml(chat.title)}</span>
      <span class="recent-chat-delete" data-delete-thread="${escapeHtml(chat.threadId)}" title="Delete recent chat" aria-label="Delete recent chat">
        <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 14h10l1-14"/><path d="M9 7V4h6v3"/></svg>
      </span>
    `;
    item.addEventListener("click", () => selectRecentChat(chat.threadId));
    const deleteBtn = item.querySelector("[data-delete-thread]");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteRecentChat(chat.threadId);
    });
    container.appendChild(item);
  }
}

async function deleteRecentChat(threadId) {
  const all = loadAllRecentChats();
  if (!all[threadId]) return;
  delete all[threadId];
  localStorage.setItem(RECENT_CHATS_KEY, JSON.stringify(all));
  renderRecentChats();
  showToast("Recent chat deleted.");
  if (threadId === state.threadId) {
    await startNewChat();
  }
}

function selectRecentChat(threadId) {
  const all = loadAllRecentChats();
  const chat = all[threadId];
  if (!chat) return;

  state.threadId = chat.threadId;
  state.chatHistory = chat.messages || [];
  state.mode = chat.mode || "Auto";
  state.dbName = chat.dbName || state.dbName;
  state.attachedDocumentIds = Array.isArray(chat.attachedDocumentIds) ? [...chat.attachedDocumentIds] : [];
  state.uploadChips = [];

  el("mode-select").value = state.mode;
  if (state.dbName) el("db-select").value = state.dbName;

  renderChatMessages();
  renderComposerExamples();
  updateComposerPlaceholder();
  renderAttachmentChips();
  switchView("chat");
  renderRecentChats();

  const lastWithMeta = [...state.chatHistory].reverse().find((m) => m.meta);
  if (lastWithMeta) updateInspector(lastWithMeta.meta);
}

// ---------------------------------------------------------------------------
// Chat send / actions
// ---------------------------------------------------------------------------
async function sendMessage(overrideText) {
  const box = el("message-box");
  const message = (overrideText !== undefined ? overrideText : box.value).trim();
  if (!message) return;

  appendMessage({ role: "user", content: message, timestamp: new Date().toISOString() });
  if (overrideText === undefined) { box.value = ""; autoResizeMessageBox(); }
  state.lastQuestion = message;
  setSendDisabled(true);

  const recentHistory = state.chatHistory
    .slice(0, -1)
    .slice(-8)
    .map((turn) => [turn.role === "user" ? "user" : "assistant", turn.content]);

  try {
    // Send the chat's attachment scope (including not-yet-ready ids) so Auto
    // can choose RAG / wait-for-indexing instead of falling through to General.
    const attachedIds = [...state.attachedDocumentIds];
    const data = await API.chat({
      message,
      thread_id: state.threadId,
      mode: state.mode,
      db_name: state.dbName,
      include_sample_data: state.includeSampleData,
      history: recentHistory,
      // When the user attached docs, scope RAG to them. Otherwise omit so
      // Auto/RAG can search all ready documents.
      document_ids: attachedIds.length > 0 ? attachedIds : undefined,
    });
    appendMessage({ role: "assistant", content: data.reply, timestamp: data.timestamp, meta: data });
    updateInspector(data);
    if ((data.mode_used || "").includes("RAG") || (data.sources || []).some((s) => s.file_name)) {
      activateTab("sources");
    }
  } catch (err) {
    appendMessage({ role: "assistant", content: `Sorry, something went wrong: ${err}`, timestamp: new Date().toISOString() });
  } finally {
    setSendDisabled(false);
  }
}

function setSendDisabled(disabled) {
  el("send-btn").disabled = disabled;
}

async function approveBuild() {
  const data = await API.approve({ thread_id: state.threadId, include_sample_data: state.includeSampleData });
  appendMessage({ role: "assistant", content: data.reply, timestamp: data.timestamp, meta: data });
  updateInspector(data);
}

async function cancelBuild() {
  const data = await API.cancel({ thread_id: state.threadId });
  appendMessage({ role: "assistant", content: data.reply, timestamp: data.timestamp, meta: data });
  updateInspector(data);
}

async function startNewChat() {
  const data = await API.newChat({ thread_id: state.threadId });
  state.threadId = data.thread_id;
  state.chatHistory = [];
  state.mode = "Auto";
  state.lastQuestion = "";
  state.attachedDocumentIds = [];
  state.uploadChips = [];
  state.dbName = null;
  el("mode-select").value = "Auto";
  // Keep the existing choice list, but force "No database selected".
  const choices = Array.from(el("db-select").options)
    .map((opt) => opt.value)
    .filter(Boolean);
  populateDbSelect(choices, null);
  state.currentSchema = { db_name: null, tables: [] };
  state.selectedSchemaTable = null;
  renderSchemaTab();
  renderChatMessages();
  renderComposerExamples();
  updateComposerPlaceholder();
  renderAttachmentChips();

  renderSqlBox("-- Ask a question in Query Database mode to see generated SQL here.");
  el("sql-badge").className = "badge badge-muted";
  el("sql-badge").textContent = "Not run";
  el("result-box").innerHTML = "<p class='empty-state'>No results yet. Run a query to see rows here.</p>";
  el("results-title").textContent = "Results";
  state.lastResult = null;
  state.currentSql = "";
  if (window.AgentHubVisualizer) window.AgentHubVisualizer.reset();
  el("debug-box").textContent = "No retries were needed yet.";
  el("sources-box").innerHTML = "<p class='empty-state'>No sources for this conversation yet.</p>";
  el("status-dot").className = "status-dot";
  el("status-text").textContent = "No query has been run yet.";
  setApproveVisible(false);

  switchView("chat");
  renderRecentChats();
}

async function refreshDatabases() {
  const data = await API.databases();
  populateDbSelect(data.choices, state.dbName);
}

async function onDbChange() {
  state.dbName = el("db-select").value || null;
  const data = await API.schema(state.dbName);
  state.currentSchema = data;
  state.selectedSchemaTable = null;
  renderSchemaTab();
}

function onModeChange() {
  state.mode = el("mode-select").value;
  renderComposerExamples();
  updateComposerPlaceholder();
  updateNavActiveStates();
}

function autoResizeMessageBox() {
  const box = el("message-box");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 140)}px`;
}

// ---------------------------------------------------------------------------
// Tools dropdown (runs an MCP placeholder action inline in the chat)
// ---------------------------------------------------------------------------
function renderToolsMenu(actions) {
  const menu = el("tools-menu");
  menu.innerHTML = actions.map((label) => `<button class="dropdown-item" data-label="${escapeHtml(label)}">${escapeHtml(label)}</button>`).join("");
  menu.querySelectorAll("[data-label]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      closeAllMenus();
      const label = btn.dataset.label;
      const data = await API.mcpRun({ label });
      appendMessage({
        role: "assistant",
        content: `🔧 **${label}**\n\n${data.output}`,
        timestamp: new Date().toISOString(),
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Databases page
// ---------------------------------------------------------------------------
async function loadDatabasesView() {
  const grid = el("databases-grid");
  grid.innerHTML = "<p class='empty-state'>Loading...</p>";
  let data;
  try {
    data = await API.databasesDetail();
  } catch (err) {
    grid.innerHTML = "<p class='empty-state'>Could not load databases.</p>";
    showToast(err.message || "Could not load databases.", "error");
    return;
  }
  if (data.databases.length === 0) {
    grid.innerHTML = "<p class='empty-state'>No databases yet. Build one from the sidebar.</p>";
    return;
  }
  grid.innerHTML = "";
  for (const db of data.databases) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-head">
        <div class="card-icon">
          <svg class="icon" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>
        </div>
        <div class="card-main">
          <div class="card-title">${escapeHtml(db.name)}</div>
          <div class="card-subtitle">Updated ${escapeHtml(new Date(db.modified).toLocaleDateString())}</div>
        </div>
        <button class="icon-btn icon-btn-sm icon-btn-danger card-delete-btn" data-action="delete" title="Delete database" aria-label="Delete ${escapeHtml(db.name)}">
          <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></svg>
        </button>
      </div>
      <div class="card-stats">
        <div><div class="card-stat-label">Tables</div><div class="card-stat-value">${db.table_count}</div></div>
        <div><div class="card-stat-label">Rows</div><div class="card-stat-value">${db.row_count.toLocaleString()}</div></div>
        <div><div class="card-stat-label">Size</div><div class="card-stat-value">${formatBytes(db.size_bytes)}</div></div>
      </div>
      <div class="card-footer">
        <button class="btn btn-primary btn-sm" data-action="query">Query this database</button>
        <button class="btn btn-ghost btn-sm" data-action="schema">Preview schema</button>
      </div>
    `;
    card.querySelector('[data-action="query"]').addEventListener("click", () => {
      state.mode = "Query Database";
      el("mode-select").value = "Query Database";
      el("db-select").value = db.name;
      state.dbName = db.name;
      switchView("chat");
      renderComposerExamples();
      el("message-box").focus();
    });
    card.querySelector('[data-action="schema"]').addEventListener("click", async () => {
      el("db-select").value = db.name;
      state.dbName = db.name;
      const schemaData = await API.schema(db.name);
      state.currentSchema = schemaData;
      state.selectedSchemaTable = null;
      switchView("chat");
      renderSchemaTab();
      activateTab("schema");
    });
    card.querySelector('[data-action="delete"]').addEventListener("click", () => {
      openDeleteDatabaseDialog(db.name);
    });
    grid.appendChild(card);
  }
}

function openDeleteDatabaseDialog(dbName) {
  state.pendingDeleteDbName = dbName;
  el("delete-db-name").textContent = dbName;
  el("delete-db-confirm-btn").disabled = false;
  el("delete-db-confirm-btn").textContent = "Delete Database";
  el("delete-db-dialog").showModal();
}

function closeDeleteDatabaseDialog() {
  state.pendingDeleteDbName = null;
  el("delete-db-dialog").close();
}

async function confirmDeleteDatabase() {
  const dbName = state.pendingDeleteDbName;
  if (!dbName) return;
  const wasSelected = state.dbName === dbName;
  const confirmBtn = el("delete-db-confirm-btn");
  confirmBtn.disabled = true;
  confirmBtn.textContent = "Deleting...";
  try {
    const data = await API.deleteDatabase(dbName);
    closeDeleteDatabaseDialog();
    const choices = data.db_choices || [];
    const previousDbName = state.dbName;
    const shouldMoveSelection = wasSelected || !choices.includes(previousDbName);
    const nextDbName = shouldMoveSelection ? data.default_db : previousDbName;
    populateDbSelect(choices, nextDbName);
    if (shouldMoveSelection || state.currentSchema?.db_name === dbName) {
      state.currentSchema = data.schema || { db_name: null, tables: [] };
      state.selectedSchemaTable = null;
      renderSchemaTab();
    }
    await loadDatabasesView();
    showToast(data.message || `Deleted ${dbName}.`);
  } catch (err) {
    confirmBtn.disabled = false;
    confirmBtn.textContent = "Delete Database";
    showToast(err.message || "Database could not be deleted.", "error");
  }
}

// ---------------------------------------------------------------------------
// In-chat RAG attachments (paperclip + chips + document manager)
// ---------------------------------------------------------------------------
function readyRagDocuments() {
  return state.ragDocuments.filter((doc) => doc.status === "ready");
}

function currentAttachmentScopeIds() {
  const readyIds = new Set(readyRagDocuments().map((d) => d.document_id));
  return state.attachedDocumentIds.filter((id) => readyIds.has(id));
}

function updateComposerPlaceholder() {
  const box = el("message-box");
  if (!box) return;
  if (state.mode === "Query Database") box.placeholder = "Ask a question about your database...";
  else if (state.mode === "Build Database") box.placeholder = "Describe the database you want to build...";
  else box.placeholder = "Ask AgentHub anything...";
}

function findRagDocument(documentId) {
  return state.ragDocuments.find((doc) => doc.document_id === documentId) || null;
}

function renderAttachmentChips() {
  const host = el("composer-attachments");
  if (!host) return;

  const chips = [];
  for (const upload of state.uploadChips) {
    chips.push({
      key: upload.localId,
      name: upload.file_name,
      status: upload.status,
      error: upload.error,
      removable: true,
      kind: "upload",
    });
  }
  for (const id of state.attachedDocumentIds) {
    if (state.uploadChips.some((u) => u.document_id === id)) continue;
    const doc = findRagDocument(id);
    chips.push({
      key: id,
      name: doc ? doc.file_name : id,
      status: doc ? doc.status : "ready",
      error: doc ? doc.error : null,
      removable: true,
      kind: "attached",
      documentId: id,
    });
  }

  if (chips.length === 0) {
    host.innerHTML = "";
    return;
  }

  host.innerHTML = chips.map((chip) => {
    let statusText = "Indexing...";
    let cls = "is-busy";
    if (chip.status === "uploading") { statusText = "Uploading..."; cls = "is-busy"; }
    else if (chip.status === "indexing" || chip.status === "processing") { statusText = "Indexing..."; cls = "is-busy"; }
    else if (chip.status === "ready" || chip.status === "already_indexed") { statusText = "✓ Ready"; cls = "is-ready"; }
    else if (chip.status === "failed") { statusText = "⚠ Failed"; cls = "is-failed"; }
    return `
      <div class="attachment-chip ${cls}" title="${escapeHtml(chip.error || chip.name)}">
        <svg class="icon" viewBox="0 0 24 24"><path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8Z"/><path d="M14 3v5h5"/></svg>
        <span class="attachment-chip-name">${escapeHtml(chip.name)}</span>
        <span class="attachment-chip-status">${escapeHtml(statusText)}</span>
        ${chip.removable ? `
          <button type="button" class="attachment-chip-remove" data-detach-key="${escapeHtml(chip.key)}" data-detach-kind="${escapeHtml(chip.kind)}" aria-label="Remove ${escapeHtml(chip.name)}">
            <svg class="icon" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>` : ""}
      </div>`;
  }).join("");

  host.querySelectorAll("[data-detach-key]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.detachKey;
      const kind = btn.dataset.detachKind;
      if (kind === "upload") {
        state.uploadChips = state.uploadChips.filter((u) => u.localId !== key);
      } else {
        // Detach from conversation scope only - do not delete from the index.
        state.attachedDocumentIds = state.attachedDocumentIds.filter((id) => id !== key);
        persistRecentChat();
      }
      renderAttachmentChips();
      renderAttachRecentList();
    });
  });
}

async function refreshRagDocuments() {
  const data = await API.ragDocuments();
  state.ragDocuments = data.documents || [];
  const readyIds = new Set(readyRagDocuments().map((d) => d.document_id));
  state.attachedDocumentIds = state.attachedDocumentIds.filter((id) => readyIds.has(id));
  renderAttachmentChips();
  renderAttachRecentList();
  if (!el("manage-docs-dialog").open) return;
  renderManageDocuments();
}

function renderAttachRecentList() {
  const list = el("attach-recent-list");
  if (!list) return;
  const recent = readyRagDocuments()
    .slice()
    .sort((a, b) => String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || "")))
    .slice(0, 8);

  if (recent.length === 0) {
    list.innerHTML = `<div class="attach-recent-empty">No indexed documents yet.</div>`;
    return;
  }

  list.innerHTML = recent.map((doc) => {
    const attached = state.attachedDocumentIds.includes(doc.document_id);
    return `
      <button type="button" class="attach-recent-item ${attached ? "is-attached" : ""}" data-attach-id="${escapeHtml(doc.document_id)}">
        <svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8Z"/><path d="M14 3v5h5"/></svg>
        <span>${escapeHtml(doc.file_name)}</span>
        <span class="meta">${attached ? "Attached" : escapeHtml((doc.file_type || "").toUpperCase())}</span>
      </button>`;
  }).join("");

  list.querySelectorAll("[data-attach-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      attachExistingDocument(btn.dataset.attachId);
      closeAllMenus();
    });
  });
}

function attachExistingDocument(documentId) {
  const doc = findRagDocument(documentId);
  if (!doc || doc.status !== "ready") {
    showToast("That document is not ready yet.", "error");
    return;
  }
  if (!state.attachedDocumentIds.includes(documentId)) {
    state.attachedDocumentIds.push(documentId);
    persistRecentChat();
  }
  renderAttachmentChips();
  renderAttachRecentList();
  showToast(`Attached ${doc.file_name}.`);
}

function openAttachMenu() {
  const menu = el("attach-menu");
  const open = menu.classList.contains("hidden");
  closeAllMenus();
  if (open) {
    renderAttachRecentList();
    menu.classList.remove("hidden");
    el("attach-btn").setAttribute("aria-expanded", "true");
  }
}

async function uploadChatFiles(files) {
  if (!files || files.length === 0) return;
  closeAllMenus();

  for (const file of files) {
    const localId = `up-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    state.uploadChips.push({
      localId,
      file_name: file.name,
      status: "uploading",
      document_id: null,
      error: null,
    });
    renderAttachmentChips();

    try {
      const chip = state.uploadChips.find((u) => u.localId === localId);
      if (chip) chip.status = "indexing";
      renderAttachmentChips();

      const record = await API.uploadRagDocument(file);
      await refreshRagDocuments();

      const idx = state.uploadChips.findIndex((u) => u.localId === localId);
      if (idx >= 0) state.uploadChips.splice(idx, 1);

      if (record.status === "failed") {
        state.uploadChips.push({
          localId,
          file_name: record.file_name || file.name,
          status: "failed",
          document_id: record.document_id || null,
          error: record.error || "Indexing failed.",
        });
        showToast(record.error || `${file.name} could not be indexed.`, "error");
      } else if (record.document_id) {
        if (!state.attachedDocumentIds.includes(record.document_id)) {
          state.attachedDocumentIds.push(record.document_id);
        }
        persistRecentChat();
        const label = record.status === "already_indexed" ? "already indexed" : "ready";
        showToast(`${record.file_name} ${label}.`);
      }
    } catch (err) {
      const chip = state.uploadChips.find((u) => u.localId === localId);
      if (chip) {
        chip.status = "failed";
        chip.error = err.message || "Upload failed.";
      }
      showToast(err.message || "Upload failed.", "error");
    }
    renderAttachmentChips();
  }

  el("chat-file-input").value = "";
  renderAttachRecentList();
}

function renderManageDocuments() {
  const list = el("manage-docs-list");
  const readyCount = readyRagDocuments().length;
  el("manage-docs-summary").textContent =
    `${readyCount} ready of ${state.ragDocuments.length} document${state.ragDocuments.length === 1 ? "" : "s"}.`;

  if (state.ragDocuments.length === 0) {
    list.innerHTML = "<p class='empty-state'>No documents indexed yet. Upload one to get started.</p>";
    return;
  }

  list.innerHTML = state.ragDocuments.map((doc) => {
    const attached = state.attachedDocumentIds.includes(doc.document_id);
    const status = doc.status === "ready" ? "Ready" : (doc.status === "failed" ? "Failed" : "Processing");
    return `
      <article class="manage-doc-row">
        <div class="manage-doc-main">
          <div class="manage-doc-title">${escapeHtml(doc.file_name)}</div>
          <div class="manage-doc-meta">${escapeHtml((doc.file_type || "").toUpperCase())} · ${formatBytes(doc.size_bytes || 0)} · ${doc.chunk_count || 0} chunks · ${status}${attached ? " · In this chat" : ""}</div>
          ${doc.error ? `<div class="manage-doc-meta" style="color:var(--danger)">${escapeHtml(doc.error)}</div>` : ""}
        </div>
        <div class="manage-doc-actions">
          <button type="button" class="btn btn-ghost btn-xs" data-manage-attach="${escapeHtml(doc.document_id)}" ${doc.status === "ready" ? "" : "disabled"}>
            ${attached ? "Attached" : "Attach"}
          </button>
          <button type="button" class="btn btn-ghost btn-xs" data-manage-delete="${escapeHtml(doc.document_id)}">Delete</button>
        </div>
      </article>`;
  }).join("");

  list.querySelectorAll("[data-manage-attach]").forEach((btn) => {
    btn.addEventListener("click", () => attachExistingDocument(btn.dataset.manageAttach));
  });
  list.querySelectorAll("[data-manage-delete]").forEach((btn) => {
    btn.addEventListener("click", () => openDeleteDocumentDialog(btn.dataset.manageDelete));
  });
}

async function openManageDocuments() {
  closeAllMenus();
  try {
    await refreshRagDocuments();
  } catch (err) {
    showToast(err.message || "Could not load documents.", "error");
  }
  renderManageDocuments();
  el("manage-docs-dialog").showModal();
}

function closeManageDocuments() {
  el("manage-docs-dialog").close();
}

function openDeleteDocumentDialog(documentId) {
  const doc = findRagDocument(documentId);
  if (!doc) return;
  state.pendingDeleteDocumentId = documentId;
  el("delete-doc-name").textContent = doc.file_name;
  el("delete-doc-confirm-btn").disabled = false;
  el("delete-doc-confirm-btn").textContent = "Delete Document";
  el("delete-doc-dialog").showModal();
}

function closeDeleteDocumentDialog() {
  state.pendingDeleteDocumentId = null;
  el("delete-doc-dialog").close();
}

async function confirmDeleteDocument() {
  const documentId = state.pendingDeleteDocumentId;
  if (!documentId) return;
  const doc = findRagDocument(documentId);
  const confirmBtn = el("delete-doc-confirm-btn");
  confirmBtn.disabled = true;
  confirmBtn.textContent = "Deleting...";
  try {
    await API.deleteRagDocument(documentId);
    state.attachedDocumentIds = state.attachedDocumentIds.filter((id) => id !== documentId);
    state.uploadChips = state.uploadChips.filter((u) => u.document_id !== documentId);
    persistRecentChat();
    closeDeleteDocumentDialog();
    await refreshRagDocuments();
    renderManageDocuments();
    showToast(doc ? `Deleted ${doc.file_name}.` : "Document deleted.");
  } catch (err) {
    confirmBtn.disabled = false;
    confirmBtn.textContent = "Delete Document";
    showToast(err.message || "Document could not be deleted.", "error");
  }
}

// ---------------------------------------------------------------------------
// Tools catalog page
// ---------------------------------------------------------------------------
const TOOL_CATEGORY_ICONS = {
  mcp: `<svg class="icon" viewBox="0 0 24 24"><path d="M10 13a5 5 0 0 0 7.1 0l2.1-2.1a5 5 0 0 0-7.1-7.1L11 4.9"/><path d="M14 11a5 5 0 0 0-7.1 0l-2.1 2.1a5 5 0 0 0 7.1 7.1L13 19.1"/></svg>`,
  local: `<svg class="icon" viewBox="0 0 24 24"><path d="M4 7h16"/><path d="M7 7V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2"/><rect x="5" y="7" width="14" height="14" rx="2"/><path d="M9 12h6M9 16h4"/></svg>`,
  data: `<svg class="icon" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>`,
  weather: `<svg class="icon" viewBox="0 0 24 24"><path d="M17.5 18H8a5 5 0 1 1 1.5-9.8A6 6 0 0 1 21 10.5 3.8 3.8 0 0 1 17.5 18Z"/></svg>`,
  time: `<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>`,
  calculator: `<svg class="icon" viewBox="0 0 24 24"><rect x="6" y="3" width="12" height="18" rx="2"/><path d="M9 7h6M9 11h.01M12 11h.01M15 11h.01M9 15h.01M12 15h.01M15 15h.01"/></svg>`,
};

const TOOL_INITIALS = {
  gmail: "GM",
  google_drive: "DR",
  google_calendar: "CA",
};

function toolStatusHtml(status) {
  const cls = status.kind || "available";
  return `<span class="tool-status ${escapeHtml(cls)}">${escapeHtml(status.label)}</span>`;
}

function categoryStatusHtml(items) {
  const connected = items.filter((item) => item.status.kind === "connected").length;
  const ready = items.filter((item) => item.status.kind === "ready").length;
  if (connected > 0) return `<span class="tool-category-status">${connected} connected</span>`;
  if (ready > 0) return `<span class="tool-category-status">${ready} ready</span>`;
  return "<span class=\"tool-category-status\">Available</span>";
}

function renderCapabilityRows(items) {
  return items.map((item) => `
    <div class="tool-card-row">
      <div class="tool-icon">${item.icon || escapeHtml(item.initials || "")}</div>
      <div class="tool-row-main">
        <div class="tool-row-top">
          <div class="tool-name">${escapeHtml(item.name)}</div>
          ${toolStatusHtml(item.status)}
        </div>
        <div class="tool-summary">${escapeHtml(item.summary)}</div>
        ${item.chips?.length ? `
          <div class="tool-card-actions">
            ${item.chips.map((chip) => `<span class="tool-chip ${chip.write ? "write" : ""}">${escapeHtml(chip.label)}</span>`).join("")}
          </div>
        ` : ""}
      </div>
    </div>
  `).join("");
}

function renderToolCategoryCard(category) {
  return `
    <section class="tool-category-card ${category.wide ? "wide" : ""}">
      <div class="tool-category-head">
        <div class="tool-category-icon">${category.icon}</div>
        <div class="tool-category-meta">
          <div class="tool-category-eyebrow">${escapeHtml(category.eyebrow)}</div>
          <h2 class="tool-category-title">${escapeHtml(category.title)}</h2>
          <p class="tool-category-copy">${escapeHtml(category.copy)}</p>
        </div>
        ${categoryStatusHtml(category.items)}
      </div>
      <div class="tool-category-body">
        ${renderCapabilityRows(category.items)}
      </div>
      ${category.footer ? `
        <div class="tool-category-footer">
          <div class="tool-footer-note">${escapeHtml(category.footer.note)}</div>
          ${category.footer.button ? `<button class="btn btn-ghost btn-sm" data-tool-action="${escapeHtml(category.footer.button.action)}">${escapeHtml(category.footer.button.label)}</button>` : ""}
        </div>
      ` : ""}
    </section>
  `;
}

function buildMcpCatalogItems(mcpData) {
  const summaries = {
    gmail: "Summarize inbox context, draft replies, and send approved messages through Gmail.",
    google_drive: "Bring Drive files into workspace flows and summarize imported documents.",
    google_calendar: "Check upcoming events and prepare approved calendar meetings.",
  };

  return (mcpData.tools || []).map((tool) => ({
    name: tool.display_name,
    initials: TOOL_INITIALS[tool.key] || tool.display_name.slice(0, 2).toUpperCase(),
    summary: summaries[tool.key] || tool.description,
    status: tool.connected ? { label: "Connected", kind: "connected" } : { label: "Connect", kind: "available" },
    chips: tool.actions.slice(0, 3).map((action) => ({ label: action.description, write: action.write_action })),
  }));
}

function localCatalogItems() {
  return [
    {
      name: "Weather",
      icon: TOOL_CATEGORY_ICONS.weather,
      summary: "Answer quick weather questions from the assistant workspace.",
      status: { label: "Ready", kind: "ready" },
      chips: [{ label: "Forecasts" }, { label: "Conditions" }],
    },
    {
      name: "Date & Time",
      icon: TOOL_CATEGORY_ICONS.time,
      summary: "Resolve dates, time zones, relative time, and scheduling language.",
      status: { label: "Ready", kind: "ready" },
      chips: [{ label: "Time zones" }, { label: "Date math" }],
    },
    {
      name: "Calculator",
      icon: TOOL_CATEGORY_ICONS.calculator,
      summary: "Handle arithmetic, percentages, comparisons, and quick quantitative checks.",
      status: { label: "Ready", kind: "ready" },
      chips: [{ label: "Math" }, { label: "Conversions" }],
    },
    {
      name: "Database Builder",
      icon: TOOL_CATEGORY_ICONS.data,
      summary: "Create local SQLite databases from plain English, with approval before creation.",
      status: { label: "Ready", kind: "ready" },
      chips: [{ label: "Schema design" }, { label: "Sample data", write: true }],
    },
    {
      name: "SQL Query Agent",
      icon: TOOL_CATEGORY_ICONS.data,
      summary: "Generate, validate, repair, and run read-only SQL against local workspace databases.",
      status: { label: "Ready", kind: "ready" },
      chips: [{ label: "Read-only SQL" }, { label: "Schema preview" }],
    },
  ];
}

async function loadToolsView() {
  const catalog = el("tools-catalog");
  catalog.innerHTML = "<p class='empty-state'>Loading...</p>";
  const mcpData = await API.mcpTools();
  const categories = [
    {
      eyebrow: "External capability catalog",
      title: "MCP Tools",
      copy: "Connectors AgentHub can route to when external services are needed. Authentication is managed separately in MCP Connections.",
      icon: TOOL_CATEGORY_ICONS.mcp,
      items: buildMcpCatalogItems(mcpData),
      footer: {
        note: "Use MCP Connections to connect Google Workspace.",
        button: { label: "Manage Connections", action: "mcp" },
      },
    },
    {
      eyebrow: "Built into this workspace",
      title: "Local Tools",
      copy: "Fast local capabilities that do not require connecting an external account.",
      icon: TOOL_CATEGORY_ICONS.local,
      items: localCatalogItems(),
      footer: {
        note: "Database tools run inside this local AgentHub workspace.",
        button: { label: "Open Chat", action: "chat" },
      },
    },
  ];

  catalog.innerHTML = categories.map(renderToolCategoryCard).join("");
  catalog.querySelectorAll("[data-tool-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.toolAction;
      if (action === "mcp") switchView("mcp");
      if (action === "chat") switchView("chat");
    });
  });
}

// ---------------------------------------------------------------------------
// MCP Connections page
// ---------------------------------------------------------------------------
async function loadMcpView() {
  const grid = el("mcp-tools-grid");
  grid.innerHTML = "<p class='empty-state'>Loading...</p>";
  const [data, integrations] = await Promise.all([API.mcpTools(), API.integrationsStatus()]);
  grid.innerHTML = "";

  const providers = document.createElement("div");
  providers.className = "integration-providers";
  providers.appendChild(renderProviderCard({
    provider: "google",
    title: "Google Workspace",
    subtitle: "Powers Gmail, Google Drive, and Google Calendar",
    icon: "G",
    status: integrations.google,
    connectHref: "/auth/google/login",
  }));
  grid.appendChild(providers);

  for (const tool of data.tools) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-head">
        <div class="card-icon">${tool.icon}</div>
        <div>
          <div class="card-title">${escapeHtml(tool.display_name)}</div>
          <div class="card-subtitle">${tool.connected ? "Connected" : "Connect the provider above to enable"}</div>
        </div>
      </div>
      <div class="card-subtitle">${escapeHtml(tool.description)}</div>
      <div class="card-actions-list"></div>
    `;
    const list = card.querySelector(".card-actions-list");
    for (const action of tool.actions) {
      const row = document.createElement("div");
      row.className = "card-action-row";
      row.innerHTML = `
        <span>${escapeHtml(action.description)}</span>
        ${action.write_action ? "<span class='write-tag'>Write</span>" : ""}
        <button class="btn btn-ghost btn-xs" ${tool.connected ? "" : "disabled"}>Run</button>
      `;
      const output = document.createElement("div");
      output.className = "settings-note";
      output.style.display = "none";
      row.querySelector("button").addEventListener("click", async () => {
        const label = `${tool.display_name}: ${action.description}`;
        const result = await API.mcpRun({ label });
        output.textContent = result.output;
        output.style.display = "block";
      });
      list.appendChild(row);
      list.appendChild(output);
    }
    grid.appendChild(card);
  }
}

function renderProviderCard({ provider, title, subtitle, icon, status, connectHref }) {
  const card = document.createElement("div");
  card.className = `integration-card ${status.connected ? "connected" : ""}`;
  const account = provider === "google"
    ? (status.account?.email || status.account?.name || "")
    : (status.account?.login || status.account?.name || "");
  const statusText = status.connected
    ? `Connected${account ? ` as ${account}` : ""}`
    : status.configured
      ? "Not connected"
      : "Missing OAuth credentials";

  card.innerHTML = `
    <div class="integration-icon">${escapeHtml(icon)}</div>
    <div class="integration-main">
      <div class="integration-title-row">
        <div>
          <div class="integration-title">${escapeHtml(title)}</div>
          <div class="integration-subtitle">${escapeHtml(subtitle)}</div>
        </div>
        <span class="badge ${status.connected ? "badge-valid" : "badge-muted"}">${escapeHtml(statusText)}</span>
      </div>
      <div class="integration-tools">${status.tools.map((tool) => `<span>${escapeHtml(tool)}</span>`).join("")}</div>
    </div>
    <div class="integration-actions">
      ${status.connected
        ? `<button class="btn btn-ghost btn-sm" data-provider="${escapeHtml(provider)}">Disconnect</button>`
        : `<a class="btn btn-primary btn-sm ${status.configured ? "" : "disabled-link"}" href="${status.configured ? connectHref : "#"}">Connect</a>`
      }
    </div>
  `;

  const disconnectBtn = card.querySelector("[data-provider]");
  if (disconnectBtn) {
    disconnectBtn.addEventListener("click", async () => {
      await API.disconnectIntegration(provider);
      loadMcpView();
    });
  }
  return card;
}

// ---------------------------------------------------------------------------
// Settings page
// ---------------------------------------------------------------------------
async function loadSettingsView() {
  const data = await API.settings();
  el("settings-model").innerHTML = `
    <div class="settings-row"><span class="label">Model</span><span class="value">${escapeHtml(data.model_name)}</span></div>
    <div class="settings-row"><span class="label">OpenAI API key</span><span class="value" style="color:${data.openai_configured ? "var(--success)" : "var(--danger)"}">${data.openai_configured ? "Configured" : "Missing"}</span></div>
    <div class="settings-row"><span class="label">Max SQL repair retries</span><span class="value">${data.max_retries}</span></div>
    <div class="settings-row"><span class="label">Conversation memory (turns)</span><span class="value">${data.max_memory_turns}</span></div>
    <div class="settings-row"><span class="label">Default database</span><span class="value">${escapeHtml(data.default_db_name)}</span></div>
  `;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  initTabs();
  initTheme();
  el("theme-switch").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  if (window.AgentHubVisualizer) window.AgentHubVisualizer.init();

  const data = await API.bootstrap();
  state.threadId = data.thread_id;
  state.dbName = null;
  state.buildExamples = data.build_examples;
  state.queryExamples = data.query_examples;
  state.unavailableModes = data.unavailable_modes;
  state.currentSchema = data.schema || { db_name: null, tables: [] };

  populateModeSelect(data.modes, data.unavailable_modes);
  populateDbSelect(data.db_choices, null);
  renderSchemaTab();
  renderComposerExamples();
  updateComposerPlaceholder();
  renderToolsMenu(data.mcp_actions);
  renderRecentChats();
  renderAttachmentChips();
  renderSqlBox("-- Ask a question in Query Database mode to see generated SQL here.");
  try { await refreshRagDocuments(); } catch (_) { /* optional on boot */ }

  // Sidebar navigation
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      if (btn.dataset.quickMode) {
        state.mode = btn.dataset.quickMode;
        el("mode-select").value = state.mode;
        renderComposerExamples();
        updateComposerPlaceholder();
      }
      switchView(view);
      if (view === "chat") el("message-box").focus();
    });
  });

  el("new-chat-btn").addEventListener("click", startNewChat);
  el("send-btn").addEventListener("click", () => sendMessage());
  el("message-box").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  el("message-box").addEventListener("input", autoResizeMessageBox);

  el("approve-btn").addEventListener("click", approveBuild);
  el("cancel-btn").addEventListener("click", cancelBuild);
  el("sample-data-checkbox").addEventListener("change", (e) => { state.includeSampleData = e.target.checked; });

  el("mode-select").addEventListener("change", onModeChange);
  el("db-select").addEventListener("change", onDbChange);
  el("schema-table-select").addEventListener("change", (e) => {
    state.selectedSchemaTable = e.target.value;
    renderSchemaTab();
  });

  el("copy-sql-btn").addEventListener("click", () => {
    if (state.currentSql) navigator.clipboard.writeText(state.currentSql);
  });

  el("export-csv-btn").addEventListener("click", () => {
    if (!state.lastResult) return;
    const { columns, rows } = state.lastResult;
    const csvLines = [columns.join(",")];
    for (const row of rows) {
      csvLines.push(row.map((cell) => `"${String(cell ?? "").replace(/"/g, '""')}"`).join(","));
    }
    downloadBlob(`query-results-${state.threadId.slice(0, 8)}.csv`, csvLines.join("\n"), "text/csv");
  });

  el("rerun-btn").addEventListener("click", () => {
    if (state.lastQuestion) sendMessage(state.lastQuestion);
  });

  el("export-btn").addEventListener("click", () => {
    const lines = state.chatHistory.map((m) => `[${timeLabel(m.timestamp)}] ${m.role === "user" ? "You" : "AgentHub Studio"}: ${m.content}`);
    downloadBlob(`agenthub-chat-${state.threadId.slice(0, 8)}.txt`, lines.join("\n\n"), "text/plain");
  });

  el("options-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    el("tools-menu").classList.add("hidden");
    el("attach-menu").classList.add("hidden");
    el("options-menu").classList.toggle("hidden");
  });
  el("tools-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    el("options-menu").classList.add("hidden");
    el("attach-menu").classList.add("hidden");
    el("tools-menu").classList.toggle("hidden");
  });
  document.addEventListener("click", (e) => {
    if (e.target.closest(".attach-wrap") || e.target.closest("#options-menu") || e.target.closest(".tools-wrap")) return;
    closeAllMenus();
  });

  el("clear-thread-btn").addEventListener("click", () => { closeAllMenus(); startNewChat(); });
  el("copy-thread-id-btn").addEventListener("click", () => {
    navigator.clipboard.writeText(state.threadId);
    closeAllMenus();
  });

  el("attach-btn").disabled = false;
  el("attach-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    openAttachMenu();
  });
  el("attach-upload-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    closeAllMenus();
    el("chat-file-input").click();
  });
  el("attach-manage-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    openManageDocuments();
  });
  el("chat-file-input").addEventListener("change", (e) => {
    uploadChatFiles(Array.from(e.target.files || []));
  });
  el("manage-docs-close-btn").addEventListener("click", closeManageDocuments);
  el("manage-docs-done-btn").addEventListener("click", closeManageDocuments);
  el("manage-docs-upload-btn").addEventListener("click", () => el("chat-file-input").click());
  el("delete-doc-cancel-btn").addEventListener("click", (e) => {
    e.preventDefault();
    closeDeleteDocumentDialog();
  });
  el("delete-doc-confirm-btn").addEventListener("click", (e) => {
    e.preventDefault();
    confirmDeleteDocument();
  });
  el("delete-doc-dialog").addEventListener("cancel", () => {
    state.pendingDeleteDocumentId = null;
  });

  el("refresh-databases-btn").addEventListener("click", loadDatabasesView);
  el("delete-db-cancel-btn").addEventListener("click", (e) => {
    e.preventDefault();
    closeDeleteDatabaseDialog();
  });
  el("delete-db-confirm-btn").addEventListener("click", (e) => {
    e.preventDefault();
    confirmDeleteDatabase();
  });
  el("delete-db-dialog").addEventListener("cancel", () => {
    state.pendingDeleteDbName = null;
  });
  el("clear-local-btn").addEventListener("click", () => {
    localStorage.removeItem(RECENT_CHATS_KEY);
    renderRecentChats();
  });

  updateNavActiveStates();
}

init();

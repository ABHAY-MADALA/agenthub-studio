// visualizer.js
// =============
// AgentHub Data Visualizer — turns the current SQL result (state.lastResult)
// into interactive charts. Never re-runs SQL; consumes the same {columns, rows}
// payload already shown in the Results panel.

(function (global) {
  "use strict";

  const LIMITS = {
    bar: 50,
    pie: 10,
    line: 200,
    area: 200,
    scatter: 500,
    kpi: 12,
  };

  const PIE_MAX_CATEGORIES = 10;

  const CHART_DEFS = [
    {
      id: "bar",
      label: "Bar Chart",
      icon: "bar",
      blurb: "Compare values across categories",
    },
    {
      id: "pie",
      label: "Pie Chart",
      icon: "pie",
      blurb: "Show parts of a whole",
    },
    {
      id: "line",
      label: "Line Chart",
      icon: "line",
      blurb: "Trends over a sequence",
    },
    {
      id: "scatter",
      label: "Scatter Plot",
      icon: "scatter",
      blurb: "Relationship between two metrics",
    },
    {
      id: "area",
      label: "Area Chart",
      icon: "area",
      blurb: "Filled trend over a sequence",
    },
    {
      id: "kpi",
      label: "KPI Cards",
      icon: "kpi",
      blurb: "Highlight key metrics",
    },
  ];

  const LIGHT_PALETTE = [
    "#2563eb",
    "#64748b",
    "#0ea5e9",
    "#475569",
    "#3b82f6",
    "#94a3b8",
    "#1d4ed8",
    "#334155",
    "#38bdf8",
    "#0f172a",
  ];
  const DARK_PALETTE = [
    "#60a5fa",
    "#a78bfa",
    "#22d3ee",
    "#f59e0b",
    "#34d399",
    "#f472b6",
    "#818cf8",
    "#fb7185",
    "#2dd4bf",
    "#facc15",
  ];

  function cssColor(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function chartTheme() {
    const dark = document.documentElement.dataset.theme === "dark";
    return {
      accent: cssColor("--accent", "#2563eb"),
      surface: cssColor("--surface", "#ffffff"),
      textSecondary: cssColor("--text-secondary", "#475569"),
      grid: dark ? "rgba(148, 163, 184, 0.24)" : "rgba(148, 163, 184, 0.18)",
      fill: dark ? "rgba(96, 165, 250, 0.2)" : "rgba(37, 99, 235, 0.15)",
      barFill: dark ? "rgba(96, 165, 250, 0.76)" : "rgba(37, 99, 235, 0.78)",
      palette: dark ? DARK_PALETTE : LIGHT_PALETTE,
    };
  }

  const MONTH_RE =
    /^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(\s+\d{2,4})?$/i;
  const ISO_MONTH_RE = /^\d{4}-\d{2}(-\d{2})?$/;
  const ID_NAME_RE = /(^id$|_id$|_pk$|^pk_|_key$)/i;
  const PCT_NAME_RE = /(percent|pct|percentage|rate|_ratio$|^ratio_)/i;

  const viz = {
    result: null,
    resultKey: null,
    chartType: null,
    xCol: null,
    yCol: null,
    aggregation: "none",
    sort: "desc",
    chart: null,
    resizeObserver: null,
    bound: false,
    truncationNote: "",
  };

  function el(id) {
    return document.getElementById(id);
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function prettifyColumn(name) {
    return String(name)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function fingerprint(result) {
    if (!result || !result.columns) return "empty";
    const cols = result.columns.join("\0");
    const rows = result.rows || [];
    const n = rows.length;
    const head = rows.slice(0, 3).map((r) => r.join("\0")).join("|");
    const tail = n > 3 ? rows[n - 1].join("\0") : "";
    return `${cols}::${n}::${head}::${tail}`;
  }

  function toNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "number") return Number.isFinite(value) ? value : null;
    if (typeof value === "boolean") return value ? 1 : 0;
    const s = String(value).trim().replace(/,/g, "");
    if (s === "" || s.toLowerCase() === "null") return null;
    if (/^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(s)) {
      const n = Number(s);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  }

  function looksLikeDate(value) {
    if (value === null || value === undefined || value === "") return false;
    if (value instanceof Date && !Number.isNaN(value.getTime())) return true;
    const s = String(value).trim();
    if (!s) return false;
    if (ISO_MONTH_RE.test(s)) return true;
    if (MONTH_RE.test(s)) return true;
    if (/^\d{4}-\d{2}-\d{2}/.test(s) || /^\d{1,2}\/\d{1,2}\/\d{2,4}/.test(s)) {
      const t = Date.parse(s);
      return !Number.isNaN(t);
    }
    return false;
  }

  function isBooleanish(value) {
    if (typeof value === "boolean") return true;
    if (value === 0 || value === 1) return true;
    if (typeof value !== "string") return false;
    const s = value.trim().toLowerCase();
    return ["true", "false", "yes", "no", "y", "n", "0", "1"].includes(s);
  }

  function analyzeColumns(result) {
    const columns = result.columns || [];
    const rows = result.rows || [];
    const metas = columns.map((name, index) => {
      const values = rows.map((row) => (row ? row[index] : null));
      const nonNull = values.filter((v) => v !== null && v !== undefined && v !== "");
      let numericCount = 0;
      let dateCount = 0;
      let boolCount = 0;
      let negativeCount = 0;
      const unique = new Set();

      for (const v of nonNull) {
        unique.add(String(v));
        const n = toNumber(v);
        if (n !== null) {
          numericCount += 1;
          if (n < 0) negativeCount += 1;
        }
        if (looksLikeDate(v)) dateCount += 1;
        if (isBooleanish(v)) boolCount += 1;
      }

      const sample = nonNull.length;
      const numericRatio = sample ? numericCount / sample : 0;
      const dateRatio = sample ? dateCount / sample : 0;
      const boolRatio = sample ? boolCount / sample : 0;
      const idLike = ID_NAME_RE.test(name);
      const mostlyUnique =
        sample > 0 && unique.size >= Math.min(sample, Math.max(8, sample * 0.9));

      let type = "categorical";
      if (sample === 0) type = "empty";
      else if (boolRatio >= 0.9 && unique.size <= 3) type = "boolean";
      else if (dateRatio >= 0.7 && numericRatio < 0.95) type = "datetime";
      else if (numericRatio >= 0.85) type = idLike && mostlyUnique ? "identifier" : "numeric";
      else if (idLike && mostlyUnique) type = "identifier";
      else type = "categorical";

      // Month labels like Jan/Feb are categorical for grouping but date-like for charts
      if (type === "categorical" && dateRatio >= 0.6) type = "datetime";

      return {
        name,
        index,
        type,
        idLike,
        uniqueCount: unique.size,
        negativeCount,
        numericRatio,
        sample,
        isNumeric: type === "numeric",
        isCategory: type === "categorical" || type === "boolean" || type === "datetime",
        isDate: type === "datetime",
        isPercent: PCT_NAME_RE.test(name),
      };
    });

    return {
      columns: metas,
      rows,
      rowCount: rows.length,
      numeric: metas.filter((c) => c.isNumeric),
      categorical: metas.filter((c) => c.isCategory),
      datetime: metas.filter((c) => c.isDate),
      identifiers: metas.filter((c) => c.type === "identifier"),
    };
  }

  function categoryCount(meta, catCol, numCol) {
    if (!catCol) return 0;
    const rows = meta.rows || (viz.result && viz.result.rows) || [];
    const set = new Set();
    for (const row of rows) {
      const key = row[catCol.index];
      if (key === null || key === undefined || key === "") continue;
      if (numCol) {
        const n = toNumber(row[numCol.index]);
        if (n === null) continue;
      }
      set.add(String(key));
    }
    return set.size;
  }

  function availabilityFor(meta) {
    const hasNumeric = meta.numeric.length > 0;
    const hasCategory = meta.categorical.length > 0 || meta.datetime.length > 0;
    const catCandidates = [...meta.datetime, ...meta.categorical];
    const bestCat = catCandidates[0] || null;
    const bestNum = meta.numeric[0] || null;
    const cats = bestCat && bestNum ? categoryCount(meta, bestCat, bestNum) : bestCat ? categoryCount(meta, bestCat, null) : 0;
    const singleValue =
      meta.rowCount === 1 && meta.numeric.length >= 1 && meta.columns.length <= 3;
    const onlyNumericScalar =
      meta.rowCount >= 1 && meta.numeric.length === 1 && meta.categorical.length === 0 && meta.datetime.length === 0;

    const pieNegative = bestNum ? bestNum.negativeCount > 0 : true;
    const pieTooMany = cats > PIE_MAX_CATEGORIES;

    return {
      bar: {
        ok: hasNumeric && hasCategory,
        reason: !hasNumeric
          ? "This result does not contain a numeric column that can be visualized."
          : !hasCategory
            ? "Bar charts need a category or date column."
            : "",
      },
      pie: {
        ok: hasNumeric && hasCategory && !pieTooMany && !pieNegative && meta.rowCount > 0,
        reason: !hasNumeric
          ? "This result does not contain a numeric column that can be visualized."
          : !hasCategory
            ? "Pie charts need a label column."
            : pieTooMany
              ? "Too many categories for a readable pie chart. Try Bar Chart instead."
              : pieNegative
                ? "Pie charts work best with non-negative values."
                : "",
      },
      line: {
        ok: hasNumeric && hasCategory,
        reason: !hasNumeric
          ? "This result does not contain a numeric column that can be visualized."
          : !hasCategory
            ? "Line charts need a category or date column for the X axis."
            : "",
      },
      scatter: {
        ok: meta.numeric.length >= 2,
        reason:
          meta.numeric.length < 2
            ? "Scatter plots need at least two numeric columns."
            : "",
      },
      area: {
        ok: hasNumeric && hasCategory,
        reason: !hasNumeric
          ? "This result does not contain a numeric column that can be visualized."
          : !hasCategory
            ? "Area charts need a category or date column for the X axis."
            : "",
      },
      kpi: {
        ok: hasNumeric,
        reason: !hasNumeric
          ? "This result does not contain a numeric column that can be visualized."
          : "",
        preferred: singleValue || onlyNumericScalar || (meta.rowCount === 1 && meta.numeric.length >= 1),
      },
    };
  }

  function recommendType(meta, availability) {
    if (availability.kpi.preferred && availability.kpi.ok) return "kpi";
    if (meta.datetime.length && meta.numeric.length && availability.line.ok) return "line";
    if (meta.numeric.length >= 2 && meta.categorical.length === 0 && availability.scatter.ok) {
      return "scatter";
    }
    if (meta.categorical.length && meta.numeric.length) {
      const cats = categoryCount(meta, meta.categorical[0] || meta.datetime[0], meta.numeric[0]);
      if (cats > 0 && cats <= 6 && availability.pie.ok) {
        // Prefer bar for comparisons; pie only when few parts-of-whole categories
        // Keep bar as primary recommendation for categorical + numeric
      }
      if (availability.bar.ok) return "bar";
      if (availability.pie.ok) return "pie";
    }
    if (availability.scatter.ok) return "scatter";
    if (availability.line.ok) return "line";
    if (availability.kpi.ok) return "kpi";
    return null;
  }

  function defaultColumns(type, meta) {
    const cats = [...meta.datetime, ...meta.categorical.filter((c) => !c.idLike), ...meta.categorical];
    const nums = meta.numeric;
    if (type === "scatter") {
      return {
        xCol: nums[0] ? nums[0].name : null,
        yCol: nums[1] ? nums[1].name : nums[0] ? nums[0].name : null,
        aggregation: "none",
        sort: "original",
      };
    }
    if (type === "kpi") {
      return {
        xCol: null,
        yCol: nums[0] ? nums[0].name : null,
        aggregation: meta.rowCount > 1 ? "sum" : "none",
        sort: "original",
      };
    }
    const x = cats[0] ? cats[0].name : null;
    const y = nums[0] ? nums[0].name : null;
    let sort = "original";
    if (type === "bar" && y) {
      const lower = y.toLowerCase();
      if (/(salary|revenue|amount|total|count|bonus|price|cost)/.test(lower)) sort = "desc";
    }
    if (type === "line" || type === "area") sort = "original";
    return { xCol: x, yCol: y, aggregation: "none", sort };
  }

  function formatNumber(colMeta, value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    const n = Number(value);
    if (colMeta && colMeta.isPercent) {
      const pct = Math.abs(n) <= 1 && !Number.isInteger(n) ? n * 100 : n;
      return `${pct.toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
    }
    const rounded = Number.isInteger(n) ? n : Math.round(n * 100) / 100;
    return rounded.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function colByName(meta, name) {
    return meta.columns.find((c) => c.name === name) || null;
  }

  function aggregateRows(points, aggregation) {
    if (aggregation === "none") return points;
    const map = new Map();
    for (const p of points) {
      const key = String(p.label);
      if (!map.has(key)) map.set(key, []);
      if (p.value !== null) map.get(key).push(p.value);
    }
    const out = [];
    for (const [label, vals] of map.entries()) {
      if (vals.length === 0) {
        out.push({ label, value: null });
        continue;
      }
      let value = null;
      if (aggregation === "sum") value = vals.reduce((a, b) => a + b, 0);
      else if (aggregation === "avg") value = vals.reduce((a, b) => a + b, 0) / vals.length;
      else if (aggregation === "count") value = vals.length;
      else if (aggregation === "min") value = Math.min(...vals);
      else if (aggregation === "max") value = Math.max(...vals);
      out.push({ label, value });
    }
    return out;
  }

  function buildCategorySeries(meta, xName, yName, aggregation, sort, limit) {
    const xCol = colByName(meta, xName);
    const yCol = colByName(meta, yName);
    if (!xCol || !yCol) return { points: [], truncated: false, total: 0 };

    let points = (viz.result.rows || []).map((row) => {
      const label = row[xCol.index];
      const value = toNumber(row[yCol.index]);
      return {
        label: label === null || label === undefined || label === "" ? "(empty)" : String(label),
        value,
      };
    }).filter((p) => p.value !== null);

    points = aggregateRows(points, aggregation);

    if (sort === "asc") points.sort((a, b) => (a.value ?? 0) - (b.value ?? 0));
    else if (sort === "desc") points.sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

    const total = points.length;
    let truncated = false;
    if (total > limit) {
      points = points.slice(0, limit);
      truncated = true;
    }
    return { points, truncated, total, xCol, yCol };
  }

  function buildScatterSeries(meta, xName, yName, limit) {
    const xCol = colByName(meta, xName);
    const yCol = colByName(meta, yName);
    if (!xCol || !yCol) return { points: [], truncated: false, total: 0 };
    const points = [];
    for (const row of viz.result.rows || []) {
      const x = toNumber(row[xCol.index]);
      const y = toNumber(row[yCol.index]);
      if (x === null || y === null) continue;
      points.push({ x, y });
    }
    const total = points.length;
    let truncated = false;
    let used = points;
    if (total > limit) {
      used = points.slice(0, limit);
      truncated = true;
    }
    return { points: used, truncated, total, xCol, yCol };
  }

  function chartIconSvg(kind) {
    const common = 'viewBox="0 0 48 48" class="viz-card-icon" aria-hidden="true"';
    if (kind === "bar") {
      return `<svg ${common}><rect x="8" y="22" width="8" height="18" rx="1.5"/><rect x="20" y="12" width="8" height="28" rx="1.5"/><rect x="32" y="18" width="8" height="22" rx="1.5"/></svg>`;
    }
    if (kind === "pie") {
      return `<svg ${common}><path d="M24 8a16 16 0 1 1-11.3 27.3L24 24Z"/><path d="M24 8v16l11.3 11.3A16 16 0 0 0 24 8Z" opacity=".45"/></svg>`;
    }
    if (kind === "line") {
      return `<svg ${common} fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 34 18 22l8 8 14-18"/></svg>`;
    }
    if (kind === "scatter") {
      return `<svg ${common}><circle cx="14" cy="30" r="2.5"/><circle cx="22" cy="18" r="2.5"/><circle cx="30" cy="26" r="2.5"/><circle cx="36" cy="14" r="2.5"/><circle cx="18" cy="12" r="2.5"/></svg>`;
    }
    if (kind === "area") {
      return `<svg ${common}><path d="M8 34 18 20l8 8 14-16v22Z" opacity=".35"/><path d="M8 34 18 20l8 8 14-16" fill="none" stroke="currentColor" stroke-width="2"/></svg>`;
    }
    return `<svg ${common} fill="none" stroke="currentColor" stroke-width="2"><path d="M12 18h24M16 26h16M20 34h8"/><circle cx="24" cy="12" r="2" fill="currentColor" stroke="none"/></svg>`;
  }

  function destroyChart() {
    if (viz.chart) {
      try {
        viz.chart.destroy();
      } catch (_) {
        /* ignore */
      }
      viz.chart = null;
    }
  }

  function setHeaderButtons(mode) {
    const back = el("viz-back-btn");
    const dl = el("viz-download-btn");
    if (back) back.classList.toggle("hidden", mode !== "chart");
    if (dl) dl.classList.toggle("hidden", mode !== "chart" || viz.chartType === "kpi");
  }

  function renderEmpty(message) {
    destroyChart();
    setHeaderButtons("empty");
    const root = el("visualize-root");
    if (!root) return;
    root.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
  }

  function renderPicker(meta, availability, recommended) {
    destroyChart();
    setHeaderButtons("picker");
    const root = el("visualize-root");
    const noNumeric = meta.numeric.length === 0;

    let html = `
      <div class="viz-intro">
        <div class="viz-intro-title">Choose a visualization</div>
        <div class="viz-intro-sub">${meta.rowCount.toLocaleString()} row${meta.rowCount === 1 ? "" : "s"} · ${meta.columns.length} column${meta.columns.length === 1 ? "" : "s"}</div>
      </div>
    `;
    if (noNumeric) {
      html += `<p class="viz-hint">This result does not contain a numeric column that can be visualized.</p>`;
    }
    html += `<div class="viz-card-grid">`;
    for (const def of CHART_DEFS) {
      const avail = availability[def.id];
      const disabled = !avail.ok;
      const rec = recommended === def.id;
      html += `
        <button type="button" class="viz-card${disabled ? " is-disabled" : ""}${rec ? " is-recommended" : ""}"
          data-viz-type="${def.id}" ${disabled ? "disabled aria-disabled='true'" : ""}
          title="${escapeHtml(avail.reason || def.blurb)}">
          <div class="viz-card-top">
            <span class="viz-card-label">${escapeHtml(def.label)}</span>
            ${rec ? '<span class="viz-rec-badge">Recommended</span>' : ""}
          </div>
          <div class="viz-card-glyph">${chartIconSvg(def.icon)}</div>
          <div class="viz-card-blurb">${escapeHtml(disabled ? avail.reason || def.blurb : def.blurb)}</div>
        </button>
      `;
    }
    html += `</div>`;
    root.innerHTML = html;
  }

  function optionsHtml(columns, selected) {
    return columns
      .map(
        (c) =>
          `<option value="${escapeHtml(c.name)}" ${c.name === selected ? "selected" : ""}>${escapeHtml(c.name)}</option>`,
      )
      .join("");
  }

  function renderConfig(meta) {
    const type = viz.chartType;
    const def = CHART_DEFS.find((d) => d.id === type);
    const cats = [...meta.datetime, ...meta.categorical];
    const nums = meta.numeric;
    let fields = "";

    if (type === "bar" || type === "line" || type === "area" || type === "pie") {
      const xLabel = type === "pie" ? "Label" : "Category";
      fields += `
        <label class="viz-field">
          <span>${xLabel}</span>
          <select id="viz-x-col" class="pill-select pill-select-sm">${optionsHtml(cats.length ? cats : meta.columns, viz.xCol)}</select>
        </label>
        <label class="viz-field">
          <span>Value</span>
          <select id="viz-y-col" class="pill-select pill-select-sm">${optionsHtml(nums, viz.yCol)}</select>
        </label>
      `;
      if (type !== "pie") {
        fields += `
          <label class="viz-field">
            <span>Aggregation</span>
            <select id="viz-agg" class="pill-select pill-select-sm">
              <option value="none" ${viz.aggregation === "none" ? "selected" : ""}>None</option>
              <option value="sum" ${viz.aggregation === "sum" ? "selected" : ""}>Sum</option>
              <option value="avg" ${viz.aggregation === "avg" ? "selected" : ""}>Average</option>
              <option value="count" ${viz.aggregation === "count" ? "selected" : ""}>Count</option>
              <option value="min" ${viz.aggregation === "min" ? "selected" : ""}>Min</option>
              <option value="max" ${viz.aggregation === "max" ? "selected" : ""}>Max</option>
            </select>
          </label>
        `;
      }
      if (type === "bar") {
        fields += `
          <label class="viz-field">
            <span>Sort</span>
            <select id="viz-sort" class="pill-select pill-select-sm">
              <option value="original" ${viz.sort === "original" ? "selected" : ""}>Original</option>
              <option value="asc" ${viz.sort === "asc" ? "selected" : ""}>Ascending</option>
              <option value="desc" ${viz.sort === "desc" ? "selected" : ""}>Descending</option>
            </select>
          </label>
        `;
      }
    } else if (type === "scatter") {
      fields += `
        <label class="viz-field">
          <span>X (numeric)</span>
          <select id="viz-x-col" class="pill-select pill-select-sm">${optionsHtml(nums, viz.xCol)}</select>
        </label>
        <label class="viz-field">
          <span>Y (numeric)</span>
          <select id="viz-y-col" class="pill-select pill-select-sm">${optionsHtml(nums, viz.yCol)}</select>
        </label>
      `;
    } else if (type === "kpi") {
      fields += `
        <label class="viz-field">
          <span>Metric</span>
          <select id="viz-y-col" class="pill-select pill-select-sm">${optionsHtml(nums, viz.yCol)}</select>
        </label>
        <label class="viz-field">
          <span>Aggregation</span>
          <select id="viz-agg" class="pill-select pill-select-sm">
            <option value="none" ${viz.aggregation === "none" ? "selected" : ""}>First value</option>
            <option value="sum" ${viz.aggregation === "sum" ? "selected" : ""}>Sum</option>
            <option value="avg" ${viz.aggregation === "avg" ? "selected" : ""}>Average</option>
            <option value="count" ${viz.aggregation === "count" ? "selected" : ""}>Count</option>
            <option value="min" ${viz.aggregation === "min" ? "selected" : ""}>Min</option>
            <option value="max" ${viz.aggregation === "max" ? "selected" : ""}>Max</option>
          </select>
        </label>
      `;
    }

    return `
      <div class="viz-config">
        <div class="viz-config-title">${escapeHtml(def ? def.label : "Chart")}</div>
        <div class="viz-config-fields">
          ${fields}
          <button type="button" id="viz-generate-btn" class="btn btn-primary btn-sm">Generate</button>
        </div>
      </div>
    `;
  }

  function tooltipLabel(colMeta, value) {
    return `${prettifyColumn(colMeta.name)}: ${formatNumber(colMeta, value)}`;
  }

  function commonChartOptions(yCol) {
    const theme = chartTheme();
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleFont: { size: 12, weight: "600" },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 6,
          callbacks: {
            label(ctx) {
              const raw = ctx.parsed.y ?? ctx.parsed ?? ctx.raw;
              const val = typeof raw === "object" && raw !== null ? raw.y : raw;
              return tooltipLabel(yCol, val);
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: theme.grid },
          ticks: { color: theme.textSecondary, maxRotation: 45, minRotation: 0, font: { size: 11 } },
        },
        y: {
          grid: { color: theme.grid },
          ticks: {
            color: theme.textSecondary,
            font: { size: 11 },
            callback(v) {
              return formatNumber(yCol, v);
            },
          },
        },
      },
    };
  }

  function renderKpi(meta) {
    const yCol = colByName(meta, viz.yCol) || meta.numeric[0];
    if (!yCol) {
      return `<p class="viz-hint">Select a numeric metric.</p>`;
    }
    const values = (viz.result.rows || [])
      .map((row) => toNumber(row[yCol.index]))
      .filter((v) => v !== null);
    let display = null;
    if (viz.aggregation === "none") display = values[0] ?? null;
    else if (viz.aggregation === "sum") display = values.reduce((a, b) => a + b, 0);
    else if (viz.aggregation === "avg") display = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    else if (viz.aggregation === "count") display = values.length;
    else if (viz.aggregation === "min") display = values.length ? Math.min(...values) : null;
    else if (viz.aggregation === "max") display = values.length ? Math.max(...values) : null;

    // Extra cards when single row and multiple numerics
    const cards = [];
    if (meta.rowCount === 1 && meta.numeric.length > 1 && viz.aggregation === "none") {
      for (const col of meta.numeric.slice(0, LIMITS.kpi)) {
        const v = toNumber(viz.result.rows[0][col.index]);
        cards.push({ title: prettifyColumn(col.name), value: formatNumber(col, v), col });
      }
    } else {
      cards.push({
        title: prettifyColumn(yCol.name),
        value: formatNumber(yCol, display),
        col: yCol,
      });
    }

    return `
      <div class="viz-kpi-grid">
        ${cards
          .map(
            (c) => `
          <div class="viz-kpi-card">
            <div class="viz-kpi-label">${escapeHtml(c.title)}</div>
            <div class="viz-kpi-value">${escapeHtml(c.value)}</div>
          </div>`,
          )
          .join("")}
      </div>
    `;
  }

  function paintChart(meta) {
    const root = el("visualize-root");
    if (!root || typeof Chart === "undefined") {
      renderEmpty("Chart library failed to load.");
      return;
    }

    setHeaderButtons("chart");
    const noteId = "viz-truncation-note";
    let body = renderConfig(meta);
    body += `<div id="${noteId}" class="viz-truncation hidden"></div>`;

    if (viz.chartType === "kpi") {
      body += renderKpi(meta);
      root.innerHTML = body;
      bindConfigHandlers(meta);
      return;
    }

    body += `
      <div class="viz-chart-shell">
        <canvas id="viz-canvas" aria-label="Visualization chart"></canvas>
      </div>
      <p id="viz-error" class="viz-hint hidden"></p>
    `;
    root.innerHTML = body;
    bindConfigHandlers(meta);

    const canvas = el("viz-canvas");
    const errEl = el("viz-error");
    const noteEl = el(noteId);
    destroyChart();

    try {
      let config = null;
      const theme = chartTheme();
      viz.truncationNote = "";

      if (viz.chartType === "scatter") {
        const series = buildScatterSeries(meta, viz.xCol, viz.yCol, LIMITS.scatter);
        if (!series.points.length) throw new Error("No numeric point pairs to plot.");
        if (series.truncated) {
          viz.truncationNote = `Showing ${series.points.length.toLocaleString()} of ${series.total.toLocaleString()} rows.`;
        }
        const opts = commonChartOptions(series.yCol);
        opts.plugins.tooltip.callbacks.label = (ctx) => {
          const p = ctx.raw;
          return [
            `${prettifyColumn(series.xCol.name)}: ${formatNumber(series.xCol, p.x)}`,
            `${prettifyColumn(series.yCol.name)}: ${formatNumber(series.yCol, p.y)}`,
          ];
        };
        opts.scales.x.title = { display: true, text: series.xCol.name, color: theme.textSecondary, font: { size: 11 } };
        opts.scales.y.title = { display: true, text: series.yCol.name, color: theme.textSecondary, font: { size: 11 } };
        config = {
          type: "scatter",
          data: {
            datasets: [
              {
                label: series.yCol.name,
                data: series.points,
                backgroundColor: theme.barFill,
                borderColor: theme.accent,
                pointRadius: 4,
                pointHoverRadius: 5,
              },
            ],
          },
          options: opts,
        };
      } else {
        const limit = LIMITS[viz.chartType] || 50;
        const series = buildCategorySeries(
          meta,
          viz.xCol,
          viz.yCol,
          viz.chartType === "pie" ? "sum" : viz.aggregation,
          viz.chartType === "bar" ? viz.sort : viz.chartType === "pie" ? "desc" : "original",
          limit,
        );
        if (!series.points.length) throw new Error("No plottable values for the selected columns.");
        if (series.truncated) {
          viz.truncationNote = `Showing ${series.points.length.toLocaleString()} of ${series.total.toLocaleString()} rows.`;
        }
        if (viz.chartType === "pie" && series.points.length > PIE_MAX_CATEGORIES) {
          throw new Error("Too many categories for a readable pie chart. Try Bar Chart instead.");
        }

        const labels = series.points.map((p) => p.label);
        const values = series.points.map((p) => p.value);
        const opts = commonChartOptions(series.yCol);

        if (viz.chartType === "pie") {
          opts.scales = {};
          opts.plugins.legend = {
            display: true,
            position: "bottom",
            labels: { boxWidth: 10, font: { size: 11 }, color: theme.textSecondary },
          };
          opts.plugins.tooltip.callbacks.title = (items) => (items[0] ? String(items[0].label) : "");
          opts.plugins.tooltip.callbacks.label = (ctx) => tooltipLabel(series.yCol, ctx.parsed);
          config = {
            type: "doughnut",
            data: {
              labels,
              datasets: [
                {
                  data: values,
                  backgroundColor: labels.map((_, i) => theme.palette[i % theme.palette.length]),
                  borderWidth: 1,
                  borderColor: theme.surface,
                  hoverOffset: 4,
                },
              ],
            },
            options: opts,
          };
        } else {
          const isArea = viz.chartType === "area";
          const isLine = viz.chartType === "line" || isArea;
          opts.plugins.tooltip.callbacks.title = (items) => (items[0] ? String(items[0].label) : "");
          config = {
            type: isLine ? "line" : "bar",
            data: {
              labels,
              datasets: [
                {
                  label: series.yCol.name,
                  data: values,
                  backgroundColor: isLine ? theme.fill : theme.barFill,
                  borderColor: theme.accent,
                  borderWidth: isLine ? 2 : 0,
                  borderRadius: isLine ? 0 : 4,
                  fill: isArea,
                  tension: isLine ? 0.25 : 0,
                  pointRadius: isLine ? 3 : 0,
                  pointHoverRadius: 5,
                },
              ],
            },
            options: opts,
          };
        }
      }

      viz.chart = new Chart(canvas.getContext("2d"), config);
      if (noteEl && viz.truncationNote) {
        noteEl.textContent = viz.truncationNote;
        noteEl.classList.remove("hidden");
      }
      ensureResizeObserver();
    } catch (err) {
      if (errEl) {
        errEl.textContent = err.message || "Unable to render this chart.";
        errEl.classList.remove("hidden");
      }
      const shell = root.querySelector(".viz-chart-shell");
      if (shell) shell.classList.add("hidden");
    }
  }

  function bindConfigHandlers(meta) {
    const gen = el("viz-generate-btn");
    if (gen) {
      gen.addEventListener("click", () => {
        readConfigFromDom();
        paintChart(meta);
      });
    }
    ["viz-x-col", "viz-y-col", "viz-agg", "viz-sort"].forEach((id) => {
      const node = el(id);
      if (!node) return;
      node.addEventListener("change", () => {
        readConfigFromDom();
        paintChart(meta);
      });
    });
  }

  function readConfigFromDom() {
    const x = el("viz-x-col");
    const y = el("viz-y-col");
    const agg = el("viz-agg");
    const sort = el("viz-sort");
    if (x) viz.xCol = x.value;
    if (y) viz.yCol = y.value;
    if (agg) viz.aggregation = agg.value;
    if (sort) viz.sort = sort.value;
  }

  function ensureResizeObserver() {
    const shell = document.querySelector(".viz-chart-shell");
    if (!shell) return;
    if (viz.resizeObserver) viz.resizeObserver.disconnect();
    viz.resizeObserver = new ResizeObserver(() => {
      if (viz.chart) {
        try {
          viz.chart.resize();
        } catch (_) {
          /* ignore */
        }
      }
    });
    viz.resizeObserver.observe(shell);
  }

  function selectChartType(type) {
    if (!viz.result) return;
    const meta = analyzeColumns(viz.result);
    const availability = availabilityFor(meta);
    if (!availability[type] || !availability[type].ok) return;
    viz.chartType = type;
    const defaults = defaultColumns(type, meta);
    viz.xCol = defaults.xCol;
    viz.yCol = defaults.yCol;
    viz.aggregation = defaults.aggregation;
    viz.sort = defaults.sort;
    paintChart(meta);
  }

  function render() {
    const root = el("visualize-root");
    if (!root) return;

    if (!viz.result || !viz.result.columns || !viz.result.columns.length) {
      renderEmpty("No data available to visualize.");
      return;
    }
    if (!viz.result.rows || viz.result.rows.length === 0) {
      renderEmpty("No data available to visualize.");
      return;
    }

    const meta = analyzeColumns(viz.result);
    const availability = availabilityFor(meta);
    const recommended = recommendType(meta, availability);

    if (!viz.chartType) {
      renderPicker(meta, availability, recommended);
      return;
    }

    // If current type became invalid after a new result, fall back to picker
    if (!availability[viz.chartType]?.ok) {
      viz.chartType = null;
      renderPicker(meta, availability, recommended);
      return;
    }

    paintChart(meta);
  }

  function onResultChange(result) {
    const key = fingerprint(result);
    if (key !== viz.resultKey) {
      destroyChart();
      viz.result = result;
      viz.resultKey = key;
      viz.chartType = null;
      viz.xCol = null;
      viz.yCol = null;
      viz.aggregation = "none";
      viz.sort = "desc";
      viz.truncationNote = "";
    } else {
      viz.result = result;
    }
    render();
  }

  function reset() {
    destroyChart();
    viz.result = null;
    viz.resultKey = null;
    viz.chartType = null;
    viz.xCol = null;
    viz.yCol = null;
    viz.aggregation = "none";
    viz.sort = "desc";
    viz.truncationNote = "";
    renderEmpty("No data available to visualize.");
  }

  function onTabShow() {
    if (viz.chart) {
      try {
        viz.chart.resize();
      } catch (_) {
        /* ignore */
      }
    }
  }

  function downloadPng() {
    if (!viz.chart) return;
    const a = document.createElement("a");
    a.href = viz.chart.toBase64Image("image/png", 1);
    a.download = `agenthub-chart-${viz.chartType || "viz"}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function bindUi() {
    if (viz.bound) return;
    viz.bound = true;
    const root = el("visualize-root");
    if (root) {
      root.addEventListener("click", (e) => {
        const card = e.target.closest("[data-viz-type]");
        if (!card || card.disabled) return;
        selectChartType(card.dataset.vizType);
      });
    }
    const back = el("viz-back-btn");
    if (back) {
      back.addEventListener("click", () => {
        destroyChart();
        viz.chartType = null;
        render();
      });
    }
    const dl = el("viz-download-btn");
    if (dl) dl.addEventListener("click", downloadPng);

    window.addEventListener("resize", () => {
      if (viz.chart) {
        try {
          viz.chart.resize();
        } catch (_) {
          /* ignore */
        }
      }
    });
    window.addEventListener("agenthub:themechange", () => {
      if (viz.result) render();
    });
  }

  // Expose pure helpers for automated tests
  const api = {
    init: bindUi,
    onResultChange,
    reset,
    onTabShow,
    render,
    _analyzeColumns: analyzeColumns,
    _availabilityFor: availabilityFor,
    _recommendType: recommendType,
    _defaultColumns: defaultColumns,
    _toNumber: toNumber,
    _LIMITS: LIMITS,
  };

  global.AgentHubVisualizer = api;
})(typeof window !== "undefined" ? window : globalThis);

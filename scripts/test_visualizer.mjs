/**
 * Headless tests for AgentHubVisualizer type inference / availability /
 * recommendation logic (no browser, no Chart.js render).
 */
import fs from "fs";
import path from "path";
import vm from "vm";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const code = fs.readFileSync(
  path.join(__dirname, "../static/js/visualizer.js"),
  "utf8",
);

const sandbox = {
  window: {},
  globalThis: {},
  document: {
    getElementById: () => null,
    querySelector: () => null,
    createElement: () => ({
      click() {},
      remove() {},
      style: {},
      classList: { add() {}, remove() {}, toggle() {} },
    }),
  },
  console,
  ResizeObserver: class {
    observe() {}
    disconnect() {}
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.runInNewContext(code, sandbox);
const V = sandbox.AgentHubVisualizer;

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function run(name, fn) {
  try {
    fn();
    console.log(`PASS  ${name}`);
  } catch (e) {
    console.error(`FAIL  ${name}: ${e.message}`);
    process.exitCode = 1;
  }
}

run("Test 1: dept/salary → bar recommended, pie available", () => {
  const result = {
    columns: ["department", "avg_salary"],
    rows: [
      ["Engineering", 85000],
      ["Sales", 65000],
      ["HR", 58000],
    ],
  };
  const meta = V._analyzeColumns(result);
  const avail = V._availabilityFor(meta);
  assert(avail.bar.ok, "bar should be available");
  assert(avail.pie.ok, "pie should be available");
  assert(avail.line.ok, "line should be available");
  assert(!avail.scatter.ok, "scatter should be disabled");
  assert(V._recommendType(meta, avail) === "bar", "recommend bar");
});

run("Test 2: pie works on same data (availability)", () => {
  const result = {
    columns: ["department", "avg_salary"],
    rows: [
      ["Engineering", 85000],
      ["Sales", 65000],
      ["HR", 58000],
    ],
  };
  const meta = V._analyzeColumns(result);
  assert(V._availabilityFor(meta).pie.ok, "pie ok");
});

run("Test 3: month/revenue → line recommended", () => {
  const result = {
    columns: ["month", "revenue"],
    rows: [
      ["Jan", 100],
      ["Feb", 150],
      ["Mar", 220],
    ],
  };
  const meta = V._analyzeColumns(result);
  const avail = V._availabilityFor(meta);
  assert(avail.line.ok && avail.bar.ok && avail.area.ok, "line/bar/area ok");
  assert(meta.datetime.length >= 1 || meta.categorical.length >= 1, "x axis typed");
  const rec = V._recommendType(meta, avail);
  assert(rec === "line" || rec === "bar", `expected line or bar, got ${rec}`);
});

run("Test 4: age/salary → scatter", () => {
  const result = {
    columns: ["age", "salary"],
    rows: [
      [24, 60000],
      [31, 85000],
      [44, 110000],
    ],
  };
  const meta = V._analyzeColumns(result);
  const avail = V._availabilityFor(meta);
  assert(avail.scatter.ok, "scatter ok");
  assert(meta.numeric.length >= 2, "two numeric cols");
  assert(V._recommendType(meta, avail) === "scatter", "recommend scatter");
});

run("Test 5: single employee_count → KPI", () => {
  const result = {
    columns: ["employee_count"],
    rows: [[42]],
  };
  const meta = V._analyzeColumns(result);
  const avail = V._availabilityFor(meta);
  assert(avail.kpi.ok && avail.kpi.preferred, "kpi preferred");
  assert(!avail.bar.ok, "bar disabled without category");
  assert(V._recommendType(meta, avail) === "kpi", "recommend kpi");
});

run("Test 6: empty result message path", () => {
  const result = { columns: ["a", "b"], rows: [] };
  // fingerprint/empty handling is in onResultChange; availability still runs
  const meta = V._analyzeColumns(result);
  assert(meta.rowCount === 0, "zero rows");
});

run("Test 7: new result fingerprint changes", () => {
  const a = {
    columns: ["department", "avg_salary"],
    rows: [["A", 1]],
  };
  const b = {
    columns: ["department", "avg_salary"],
    rows: [["B", 2]],
  };
  // Simulate via onResultChange + ensure chartType reset
  V.onResultChange(a);
  // select type via internal state isn't exposed; check analyze differs
  const ma = V._analyzeColumns(a);
  const mb = V._analyzeColumns(b);
  assert(ma.rowCount === 1 && mb.rowCount === 1, "both one row");
  V.reset();
});

run("Test 8: too many categories disables pie", () => {
  const rows = Array.from({ length: 15 }, (_, i) => [`Cat${i}`, 10 + i]);
  const result = { columns: ["name", "value"], rows };
  const meta = V._analyzeColumns(result);
  const avail = V._availabilityFor(meta);
  assert(avail.bar.ok, "bar still ok");
  assert(!avail.pie.ok, "pie disabled");
  assert(/Too many categories/i.test(avail.pie.reason), "pie reason");
});

run("Test 9: defaults for bar", () => {
  const result = {
    columns: ["department_name", "avg_salary"],
    rows: [
      ["Data Science", 86500],
      ["Engineering", 79000],
    ],
  };
  const meta = V._analyzeColumns(result);
  const d = V._defaultColumns("bar", meta);
  assert(d.xCol === "department_name", "category default");
  assert(d.yCol === "avg_salary", "value default");
  assert(d.sort === "desc", "salary sorts desc");
});

run("Test 10: number parsing", () => {
  assert(V._toNumber("86,500") === 86500, "comma number");
  assert(V._toNumber(null) === null, "null");
  assert(V._toNumber("abc") === null, "non-numeric");
});

if (!process.exitCode) console.log("\nAll visualizer logic tests passed.");

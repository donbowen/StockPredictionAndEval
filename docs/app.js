/* Static (GitHub Pages) version of streamlit_app.py.
 *
 * Data comes from docs/data/*.json, written by build_static_site.py. Anything that depends on a widget is
 * computed here, with the same formulas as the Streamlit app. Page text is in index.html.
 */
"use strict";

// ---------------------------------------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------------------------------------
const cache = {};
function load(name) {
  if (!cache[name]) {
    cache[name] = fetch(`data/${name}.json`).then(r => {
      if (!r.ok) throw new Error(`data/${name}.json: ${r.status}`);
      return r.json();
    });
  }
  return cache[name];
}

// ---------------------------------------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------------------------------------
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

function slug(text) {
  return text.trim().replace(/ /g, "-").replace(/[()]/g, "").toLowerCase();
}

// Markdown with $...$ math (KaTeX) and \$ for a literal dollar sign, like st.markdown.
function renderMarkdown(src) {
  const math = [];
  let s = src.replace(/\\\$/g, "\uE000DOLLAR\uE000");
  s = s.replace(/\$([^$\n]+?)\$/g, (_, tex) => {
    math.push(tex);
    return `\uE000MATH${math.length - 1}\uE000`;
  });
  let html = marked.parse(s, { gfm: true });
  html = html.replace(/\uE000MATH(\d+)\uE000/g, (_, i) => {
    try { return katex.renderToString(math[+i], { throwOnError: false }); } catch (e) { return math[+i]; }
  });
  return html.replace(/\uE000DOLLAR\uE000/g, "$");
}

function dedent(text) {
  const lines = text.replace(/^\n+/, "").replace(/\s+$/, "").split("\n");
  const indents = lines.filter(l => l.trim()).map(l => l.match(/^ */)[0].length);
  const n = indents.length ? Math.min(...indents) : 0;
  return lines.map(l => l.slice(n)).join("\n");
}

function md(textOrId, vars = {}) {
  let src = textOrId.startsWith("#md-") ? document.getElementById(textOrId.slice(1)).textContent : textOrId;
  src = dedent(src).replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
  const div = el("div", { class: "md", html: renderMarkdown(src) });
  div.querySelectorAll("h1,h2,h3,h4").forEach(h => { if (!h.id) h.id = slug(h.textContent); });
  div.querySelectorAll("pre code").forEach(c => { if (window.hljs) hljs.highlightElement(c); });
  div.querySelectorAll("a[href^='http']").forEach(a => { a.target = "_blank"; a.rel = "noopener"; });
  return div;
}

const callout = (kind, ...content) => el("div", { class: `callout ${kind}` }, ...content);
const caption = text => el("p", { class: "caption" }, text);
const columns = (...cols) => el("div", { class: "columns" }, ...cols.map(c => el("div", { class: "col" }, c)));

// ---------------------------------------------------------------------------------------------------------
// Widgets. Each keeps its value in `state[key]` so it survives re-rendering.
// ---------------------------------------------------------------------------------------------------------
const state = {};
function remember(key, dflt) {
  if (!(key in state)) state[key] = dflt;
  return state[key];
}

function selectbox(label, options, key, dflt, onChange) {
  const value = remember(key, dflt ?? options[0]);
  const sel = el("select", { onchange: e => { state[key] = options[e.target.selectedIndex]; onChange(state[key]); } },
    options.map(o => el("option", { selected: String(o) === String(value) }, String(o))));
  return el("label", { class: "widget" }, el("span", { class: "widget-label" }, label), sel);
}

function radio(label, options, key, dflt, onChange) {
  const value = remember(key, dflt ?? options[0]);
  const name = `radio-${key}`;
  return el("div", { class: "widget" }, el("span", { class: "widget-label" }, label),
    el("div", { class: "radio-row" }, options.map(o => el("label", { class: "radio" },
      el("input", { type: "radio", name, checked: o === value, onchange: () => { state[key] = o; onChange(o); } }),
      " ", String(o)))));
}

function slider(label, min, max, step, key, dflt, onChange) {
  const value = remember(key, dflt ?? min);
  const out = el("span", { class: "slider-value" }, String(value));
  const input = el("input", { type: "range", min, max, step, value,
    oninput: e => { out.textContent = e.target.value; },
    onchange: e => { state[key] = +e.target.value; onChange(state[key]); } });
  return el("label", { class: "widget" }, el("span", { class: "widget-label" }, label),
    el("div", { class: "slider-row" }, input, out));
}

function numberInput(label, min, max, step, key, dflt, onChange) {
  const value = remember(key, dflt);
  const input = el("input", { type: "number", min, max, step, value,
    onchange: e => {
      let v = parseFloat(e.target.value);
      if (!isFinite(v)) v = dflt;
      v = Math.min(max, Math.max(min, v));
      e.target.value = v;
      state[key] = v;
      onChange(v);
    } });
  return el("label", { class: "widget" }, el("span", { class: "widget-label" }, label), input);
}

// Chip multiselect with a type-to-filter dropdown (st.multiselect)
function multiselect(label, options, key, dflt, onChange) {
  const selected = remember(key, [...dflt]);
  const wrap = el("div", { class: "widget multiselect" });
  const chips = el("div", { class: "chips" });
  const input = el("input", { type: "text", placeholder: "Choose an option" });
  const list = el("div", { class: "ms-list" });
  const box = el("div", { class: "ms-box" }, chips, input);
  wrap.append(el("span", { class: "widget-label" }, label), box, list);

  function changed() { drawChips(); drawList(); onChange(state[key]); }
  function drawChips() {
    chips.replaceChildren(...state[key].map(o => el("span", { class: "chip" }, o,
      el("button", { class: "chip-x", title: "Remove", onclick: e => {
        e.stopPropagation(); state[key] = state[key].filter(x => x !== o); changed();
      } }, "×"))));
    if (state[key].length) chips.append(el("button", { class: "chip-clear", title: "Clear all", onclick: e => {
      e.stopPropagation(); state[key] = []; changed();
    } }, "clear"));
  }
  function drawList() {
    const q = input.value.toLowerCase();
    const avail = options.filter(o => !state[key].includes(o) && o.toLowerCase().includes(q));
    list.replaceChildren(...(avail.length ? avail.map(o => el("div", { class: "ms-item", onmousedown: e => {
      e.preventDefault(); state[key] = [...state[key], o]; input.value = ""; changed();
    } }, o)) : [el("div", { class: "ms-empty" }, "No results")]));
  }
  box.addEventListener("click", () => input.focus());
  input.addEventListener("focus", () => { drawList(); wrap.classList.add("open"); });
  input.addEventListener("blur", () => wrap.classList.remove("open"));
  input.addEventListener("input", drawList);
  drawChips();
  return wrap;
}

// Tabs that render their content the first time they are shown, and remember the open tab.
function tabs(key, specs) {
  const active = remember(key, 0);
  const bar = el("div", { class: "tab-bar" });
  const panels = el("div", { class: "tab-panels" });
  const rendered = [];
  specs.forEach(([name, render], i) => {
    const panel = el("div", { class: "tab-panel", hidden: i !== active });
    const btn = el("button", { class: "tab" + (i === active ? " active" : ""), onclick: () => {
      state[key] = i;
      bar.querySelectorAll(".tab").forEach((b, j) => b.classList.toggle("active", j === i));
      panels.querySelectorAll(":scope > .tab-panel").forEach((p, j) => { p.hidden = j !== i; });
      show(i);
    } }, name);
    bar.append(btn);
    panels.append(panel);
  });
  function show(i) {
    const panel = panels.children[i];
    if (!rendered[i]) {
      rendered[i] = true;
      Promise.resolve(specs[i][1](panel)).catch(err => panel.append(callout("error", String(err))));
    } else {
      panel.querySelectorAll(".js-plotly-plot").forEach(p => Plotly.Plots.resize(p));
    }
  }
  setTimeout(() => show(active), 0);
  return el("div", { class: "tabs" }, bar, panels);
}

// A re-renderable region: region(fn) calls fn(div) now and whenever .refresh() is called.
function region(fn) {
  const div = el("div");
  div.refresh = () => { div.replaceChildren(); Promise.resolve(fn(div)).catch(err => div.append(callout("error", String(err)))); };
  queueMicrotask(div.refresh);   // after the caller has set up the widgets (and their defaults) this region reads
  return div;
}

// ---------------------------------------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------------------------------------
const fmt = {
  f: d => v => (v === null || v === undefined || Number.isNaN(v)) ? "" : v.toFixed(d),
  pct: d => v => (v === null || Number.isNaN(v)) ? "" : (100 * v).toFixed(d) + "%",
  int: v => (v === null || v === undefined || Number.isNaN(v)) ? "" : Math.round(v).toLocaleString("en-US"),
  auto: v => {
    if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "None";
    if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(4);
    return String(v);
  },
};

/* rows: [{label, values: {col: v}}] ; formats: {col: fn} ; opts.rowClass(label) */
function dataTable(cols, rows, { formats = {}, indexName = "", rowClass, defaultFormat = fmt.f(1), maxHeight } = {}) {
  const thead = el("thead", {}, el("tr", {}, el("th", {}, indexName), cols.map(c => el("th", {}, c))));
  const tbody = el("tbody", {}, rows.map(r => el("tr", { class: rowClass ? rowClass(r.label) : null },
    el("th", {}, r.label),
    cols.map(c => el("td", {}, (formats[c] || defaultFormat)(r.values[c]))))));
  return el("div", { class: "table-wrap", style: maxHeight ? { maxHeight: maxHeight + "px" } : null },
    el("table", { class: "data" }, thead, tbody));
}

// Raw rows (array of arrays) with a numeric index column, like st.dataframe of a DataFrame
function frameTable({ columns, rows }, { rowStyle, maxHeight, index = true } = {}) {
  const thead = el("thead", {}, el("tr", {}, index ? el("th", {}, "") : null, columns.map(c => el("th", {}, c))));
  const tbody = el("tbody", {}, rows.map((r, i) => el("tr", { style: rowStyle ? rowStyle(r, i) : null },
    index ? el("th", {}, String(i)) : null, r.map(v => el("td", { class: typeof v === "string" ? "txt" : null }, fmt.auto(v))))));
  return el("div", { class: "table-wrap", style: maxHeight ? { maxHeight: maxHeight + "px" } : null },
    el("table", { class: "data" }, thead, tbody));
}

// Alpha / t-stat table ("Table 1"): alphas to 3 decimals, (t-stats) to 2, significant t-stats bold.
function alphaTable(t, colNames = t.cols) {
  const tbody = el("tbody");
  t.rows.forEach(([model, metric], i) => {
    const tr = el("tr", { class: metric === "t-stat" ? "tstat" : "alpha" });
    if (metric === "alpha") tr.append(el("th", { rowspan: 2, class: "lvl0" }, model));
    tr.append(el("th", { class: "lvl1" }, metric));
    t.values[i].forEach(v => {
      const sig = metric === "t-stat" && Math.abs(v) > 1.96;
      tr.append(el("td", { class: sig ? "sig" : null },
        v === null ? "" : metric === "alpha" ? v.toFixed(3) : `(${v.toFixed(2)})`));
    });
    tbody.append(tr);
  });
  return el("div", { class: "table-wrap" }, el("table", { class: "alpha-table" },
    el("thead", {}, el("tr", {}, el("th", { class: "lvl0" }, ""), el("th", { class: "lvl1" }, ""),
      colNames.map(c => el("th", {}, c)))), tbody));
}

// ---------------------------------------------------------------------------------------------------------
// Statistics (pandas conventions: NaN is skipped, std uses n - 1)
// ---------------------------------------------------------------------------------------------------------
const isNum = v => v !== null && v !== undefined && !Number.isNaN(v);
function mean(a) { let s = 0, n = 0; for (const v of a) if (isNum(v)) { s += v; n++; } return n ? s / n : NaN; }
function std(a) {
  const m = mean(a); let s = 0, n = 0;
  for (const v of a) if (isNum(v)) { s += (v - m) ** 2; n++; }
  return n > 1 ? Math.sqrt(s / (n - 1)) : NaN;
}
const sub = (a, b) => a.map((v, i) => (isNum(v) && isNum(b[i])) ? v - b[i] : NaN);

// cumprod of (1 + r) skipping NaN, with a leading 1
function wealth(r) {
  const out = [1]; let w = 1;
  for (const v of r) { if (isNum(v)) { w *= 1 + v; out.push(w); } else out.push(NaN); }
  return out;
}

function returnStats(r, rf = null) {
  const excess = rf ? sub(r, rf) : r;
  const w = wealth(r);
  let peak = -Infinity, mdd = 0;
  for (const v of w) { if (!isNum(v)) continue; peak = Math.max(peak, v); mdd = Math.max(mdd, 1 - v / peak); }
  return {
    "Mean (%/yr)": 1200 * mean(r),
    "Std. dev. (%/yr)": 100 * Math.sqrt(12) * std(r),
    "Sharpe ratio": Math.sqrt(12) * mean(excess) / std(excess),
    "Max drawdown (%)": 100 * mdd,
  };
}

// Least squares with ordinary and Newey-West (Bartlett, no small-sample correction, as statsmodels) errors
function ols(y, X, hacLags = null) {
  const n = y.length, k = X[0].length;
  const XtX = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => {
    let s = 0; for (let t = 0; t < n; t++) s += X[t][i] * X[t][j]; return s;
  }));
  const Xty = Array.from({ length: k }, (_, i) => { let s = 0; for (let t = 0; t < n; t++) s += X[t][i] * y[t]; return s; });
  const inv = invert(XtX);
  const b = inv.map(row => row.reduce((s, v, j) => s + v * Xty[j], 0));
  const u = y.map((v, t) => v - X[t].reduce((s, x, j) => s + x * b[j], 0));
  let cov;
  if (hacLags === null) {
    const s2 = u.reduce((s, v) => s + v * v, 0) / (n - k);
    cov = inv.map(row => row.map(v => v * s2));
  } else {
    const xu = X.map((row, t) => row.map(x => x * u[t]));
    const S = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let lag = 0; lag <= hacLags; lag++) {
      const w = lag === 0 ? 1 : 1 - lag / (hacLags + 1);
      for (let t = lag; t < n; t++) for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        const g = xu[t][i] * xu[t - lag][j];
        S[i][j] += lag === 0 ? g : w * g;
        if (lag > 0) S[j][i] += w * g;
      }
    }
    cov = matmul(matmul(inv, S), inv);
  }
  return { params: b, tvalues: b.map((v, i) => v / Math.sqrt(cov[i][i])) };
}
function matmul(A, B) { return A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0))); }
function invert(M) {
  const n = M.length, A = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => +(i === j))]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map(r => r.slice(n));
}

// Factors (percent per month) looked up by date
function factorLookup(main) {
  const f = main.factors, pos = new Map(f.dates.map((d, i) => [d, i]));
  const get = (col, ds) => ds.map(d => pos.has(d) ? f[col][pos.get(d)] : NaN);
  return {
    get,
    rf: ds => get("RF", ds).map(v => v / 100),
    market: ds => get("mkt_excess", ds).map((v, i) => v / 100 + get("RF", [ds[i]])[0] / 100),
  };
}

const yearOf = d => +d.slice(0, 4);
function since(series, start, cols) {
  const keep = series.dates.map(d => yearOf(d) >= start);
  const out = { dates: series.dates.filter((_, i) => keep[i]) };
  for (const c of cols) out[c] = series[c].filter((_, i) => keep[i]);
  return out;
}

// ---------------------------------------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------------------------------------
const PLOT_CONFIG = { responsive: true, displaylogo: false };
function plot(container, traces, layout) {
  const div = el("div", { class: "plot" });
  container.append(div);
  Plotly.newPlot(div, traces, Object.assign({ margin: { l: 60, r: 20, t: 50, b: 40 } }, layout), PLOT_CONFIG);
  return div;
}
function monthBefore(d) {
  const [y, m] = d.split("-").map(Number);
  const py = m === 1 ? y - 1 : y, pm = m === 1 ? 12 : m - 1;
  return `${py}-${String(pm).padStart(2, "0")}-${d.slice(8)}`;
}
const VIRIDIS = { 0: "#440154", 0.2: "#414487", 0.4: "#2a788e", 0.6: "#22a884", 0.8: "#7ad151", 1: "#fde725" };
const BIN_STYLE = b => ({
  color: VIRIDIS[Math.round((1.2 - b / 5) * 10) / 10],
  dash: b === 1 ? "solid" : b === 2 ? "dash" : b === 3 ? "dashdot" : "dot",
});
const logTicks = max => [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].filter(v => v <= max * 2);

function binTraces(main, model, values, extra = {}) {
  return ["1", "2", "3", "4", "5", "LS"].map(b => {
    const ls = b === "LS";
    const s = ls ? { color: VIRIDIS[0], dash: "solid" } : BIN_STYLE(+b);
    return Object.assign({
      name: ls ? "Long-Short" : `Portfolio ${b}`, mode: "lines", y: values(main.returns[model][b]),
      line: { color: s.color, dash: s.dash, width: ls ? 3 : 1.5 }, opacity: ls ? 1 : 0.7,
    }, extra);
  });
}

// ---------------------------------------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------------------------------------
const PAGES = [
  ["intro", "Introduction", introPage],
  ["background", "Background & Methods", backgroundPage],
  ["compare", "Compare Models", comparePage],
  ["details", "Dig into a Model", detailsPage],
  ["practice", "Trading the Model in Practice", practicePage],
];

async function introPage(page) {
  page.append(el("h1", {}, "📊 Stock Prediction Dashboard (Lehigh FIN377)"), md("#md-intro"),
    callout("success", md("#md-intro-success")));
  const main = await load("main");
  page.append(el("ul", {}, Object.entries(main.modelIndex).map(([k, v]) => el("li", {}, el("strong", {}, k), ": ", v.description))));
  page.append(callout("info", `Data loaded successfully. Found ${main.models.length} models with out-of-sample performance for the period from ${main.dates[0]} to ${main.dates[main.dates.length - 1]}.`));
  page.append(el("h3", {}, "Available Models"), el("p", {}, [...main.models].sort().join(", ")));
}

// ----- Compare Models --------------------------------------------------------------------------------------
async function comparePage(page) {
  page.append(el("h1", {}, "Compare Model Performance"));
  const main = await load("main");
  const defaults = ["OLS", "LassoCV", "MLP-rank-256-128-64", "HGB-return", "HGB-quintile"].filter(m => main.models.includes(m));
  const body = region(div => {
    const models = state.compare_models;
    if (!models.length) { div.append(callout("warning", "Please select at least one model to continue.")); return; }
    div.append(tabs("compare_tab", [
      ["Cumulative Returns Comparison", p => compareCumulative(p, main, models)],
      ["Risk-Adjusted Returns via Regression Analysis", p => compareFactors(p, main, models)],
      ["Additional Metrics", p => compareMetrics(p, main, models)],
      ["Models on This Site", p => modelsOnThisSite(p, main)],
    ]));
  });
  page.append(multiselect("Select models to compare:", main.models, "compare_models", defaults, () => body.refresh()), body);
}

function compareCumulative(p, main, models) {
  p.append(el("h2", {}, "Long-Short Portfolio Returns Comparison"), md("#md-compare-cum"));
  const x = [monthBefore(main.dates[0]), ...main.dates];
  const series = models.map(m => ({ m, y: wealth(main.returns[m].LS) }));
  const all = series.flatMap(s => s.y.filter(isNum));
  const max = Math.max(...all), min = Math.min(...all);
  const log = max > 50;
  const colors = main.modelColors;
  const traces = series.map(s => ({ x, y: s.y, name: s.m, mode: "lines", line: { color: colors[s.m], width: 2.2 },
    hovertemplate: "%{y:,.1f}" }));
  // Label each line at its end, nudged apart so labels don't overlap
  const tr = v => log ? Math.log10(v) : v;
  const gap = 0.045 * (tr(max) - tr(min));
  const ends = series.map(s => ({ m: s.m, pos: tr(s.y[s.y.length - 1]) })).sort((a, b) => a.pos - b.pos);
  for (let i = 1; i < ends.length; i++) ends[i].pos = Math.max(ends[i].pos, ends[i - 1].pos + gap);
  const annotations = ends.map(e => ({ x: x[x.length - 1], y: e.pos, text: `<b>${e.m}</b>`, showarrow: false,
    xanchor: "left", xshift: 8, font: { color: colors[e.m], size: 13 } }));
  const yaxis = { title: "Value of $1 invested", type: log ? "log" : "linear" };
  if (log) { const t = logTicks(max); yaxis.tickvals = t; yaxis.ticktext = t.map(v => v.toLocaleString("en-US")); }
  plot(p, traces, { height: 600, showlegend: false, hovermode: "x unified", margin: { l: 70, r: 190, t: 20, b: 30 },
    yaxis, xaxis: { title: "" }, annotations });
}

function compareFactors(p, main, models) {
  p.append(el("h2", {}, "Factor Analysis Comparison"), md("#md-compare-factor"));
  const first = main.table1[models[0]];
  const ls = m => main.table1[m].cols.indexOf("binLS");
  const t = { rows: first.rows, cols: models, values: first.rows.map((_, i) => models.map(m => main.table1[m].values[i][ls(m)])) };
  p.append(alphaTable(t), md("#md-regression-notes"));
}

function costRows(costs, model, universe, start) {
  const d = costs.data[model]?.[universe];
  if (!d) return [null, 0];
  const s = since(d, start, ["ls_gross", "ls_net", "long_gross", "turnover_long", "turnover_short", "short_illiquid"]);
  if (!s.dates.length) return [null, 0];
  return [s, +(s.dates[0] === costs.dates[0])];
}

function longShortMetrics(costs, models, universe, start, returns) {
  const rows = [];
  for (const m of models) {
    const [d, skip] = costRows(costs, m, universe, start);
    if (!d) continue;
    rows.push({ label: m, values: { ...returnStats(returns(d)),
      "Turnover long (%/mo)": 100 * mean(d.turnover_long.slice(skip)),
      "Turnover short (%/mo)": 100 * mean(d.turnover_short.slice(skip)),
      "Short in illiquid (%)": 100 * mean(d.short_illiquid) } });
  }
  return rows;
}
const LS_COLS = ["Mean (%/yr)", "Std. dev. (%/yr)", "Sharpe ratio", "Max drawdown (%)", "Turnover long (%/mo)", "Turnover short (%/mo)", "Short in illiquid (%)"];
const LO_COLS = ["Mean (%/yr)", "Std. dev. (%/yr)", "Sharpe ratio", "Max drawdown (%)", "Turnover (%/mo)", "Mean minus market (%/yr)"];
const metricsTable = (cols, rows) => dataTable(cols, rows, { formats: { "Sharpe ratio": fmt.f(2) } });

async function compareMetrics(p, main, models) {
  p.append(el("h2", {}, "Additional Performance Metrics about the Long-Short Portfolios"));
  const costs = await load("costs");
  const F = factorLookup(main);
  const body = region(div => {
    const start = state.metrics_start;
    const hs = state.liquid_half_spread / 10000, fee = state.borrow_fee / 100;
    div.append(md("#md-metrics-gross"));
    div.append(metricsTable(LS_COLS, longShortMetrics(costs, models, "All stocks", start, d => d.ls_gross)));
    div.append(el("h3", {}, "Performance with transaction costs"), md("#md-metrics-net"));
    div.append(metricsTable(LS_COLS, longShortMetrics(costs, models, "All stocks", start, d => d.ls_net)));
    div.append(el("h3", {}, "Performance with cost mitigation"), md("#md-metrics-liquid"));
    div.append(columns(
      numberInput("Half-spread, each way (basis points)", 0, 100, 1, "liquid_half_spread", 5, () => body.refresh()),
      numberInput("Borrowing fee on the short leg (% per year)", 0, 10, 0.25, "borrow_fee", 0.25, () => body.refresh())));
    div.append(caption("Spreads were wider before decimal prices (2001), so a few basis points understates costs in " +
      "2000-2002. Most stocks cost less than 1% a year to borrow, and large ones typically about " +
      "0.25% (D'Avolio, 2002). Stocks that are heavily shorted can cost much more."));
    div.append(metricsTable(LS_COLS, longShortMetrics(costs, models, "Most liquid 20%", start,
      d => d.ls_gross.map((g, i) => g - 2 * hs * (d.turnover_long[i] + d.turnover_short[i]) - fee / 12))));
    div.append(el("h3", {}, "Long-only performance"), md("#md-metrics-long"));
    const rows = [];
    let months = null;
    for (const m of models) {
      const [d, skip] = costRows(costs, m, "Most liquid 20%", start);
      if (!d) continue;
      const r = d.long_gross.map((g, i) => g - 2 * hs * d.turnover_long[i]);
      rows.push({ label: m, values: { ...returnStats(r, F.rf(d.dates)), "Turnover (%/mo)": 100 * mean(d.turnover_long.slice(skip)),
        "Mean minus market (%/yr)": 1200 * mean(sub(r, F.market(d.dates))) } });
      months = d.dates;
    }
    if (months) rows.push({ label: "Stock market", values: { ...returnStats(F.market(months), F.rf(months)), "Mean minus market (%/yr)": 0 } });
    div.append(metricsTable(LO_COLS, rows));
  });
  remember("liquid_half_spread", 5); remember("borrow_fee", 0.25);
  p.append(radio("Sample starts in:", [2000, 2005, 2010], "metrics_start", 2000, () => body.refresh()), body);
}

function modelsOnThisSite(p, main) {
  p.append(el("h2", {}, "Models on This Site"), md("#md-models-intro"));
  const cols = ["Family", "Predicts", "Tuning", "What it is"];
  const head = { Family: el("a", { href: "#about-the-estimators" }, "Estimator Family"),
    Predicts: el("a", { href: "#what-is-the-target" }, "Predicts (Target)") };
  const table = el("table", { class: "model-guide" },
    el("thead", {}, el("tr", {}, el("th", {}, "Model"), cols.map(c => el("th", {}, head[c] || c)))),
    el("tbody", {}, Object.entries(main.modelIndex).map(([name, info]) => {
      const g = main.modelGuide[name] || { "What it is": info.description };
      return el("tr", {}, el("td", {}, name), cols.map(c => el("td", {}, g[c] || "")));
    })));
  p.append(table, md("#md-models-designs"));
  p.append(el("h2", { id: "about-the-estimators" }, "About the Estimators"), md("#md-estimators"));
  p.append(el("h3", { id: "what-is-the-target", class: "divider" }, "What is the target?"), md("#md-target"));
}

// ----- Dig into a Model ------------------------------------------------------------------------------------
async function detailsPage(page) {
  page.append(el("h1", {}, "Detailed Model Analysis"));
  const main = await load("main");
  const names = ["LassoCV", ...[...main.models].filter(m => m !== "LassoCV").sort()];
  const body = region(div => {
    const m = state.details_model;
    div.append(tabs("details_tab", [
      ["Cumulative Returns", p => detailsCumulative(p, main, m)],
      ["Against the Factor Zoo", p => detailsZoo(p, main, m)],
      ["Moving Average Returns", p => detailsMovingAverage(p, main, m)],
      ["Regression Analysis", p => detailsRegression(p, main, m)],
      ["Model Statistics", p => detailsStatistics(p, main, m)],
      ["Turnover Patterns", p => detailsTurnover(p, main, m)],
      ["Model Pipeline", p => detailsPipeline(p, m)],
      ["Variable Interpretation", p => detailsImportance(p, m)],
    ]));
  });
  page.append(selectbox("Select a model to analyze:", names, "details_model", "LassoCV", () => body.refresh()), body);
}

function detailsCumulative(p, main, m) {
  p.append(el("h2", {}, "Cumulative Returns"), md("#md-details-cum"));
  const x = [monthBefore(main.dates[0]), ...main.dates];
  const traces = binTraces(main, m, r => wealth(r), { x });
  const max = Math.max(...traces.flatMap(t => t.y.filter(isNum)));
  const yaxis = max > 50 ? { type: "log", tickformat: ".0f" } : { type: "linear" };
  plot(p, traces, { title: `Cumulative Returns for ${m} Model`, height: 550, yaxis, xaxis: { title: "Date" },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.7)" }, hovermode: "x unified" });
}

async function detailsZoo(p, main, m) {
  p.append(el("h2", {}, "Cumulative Returns vs. Anomaly Zoo"), md("#md-details-zoo"));
  const zoo = await load("zoo");
  const chart = region(div => {
    let chosen = state.zoo_returns.includes("Select All") || !state.zoo_returns.length ? zoo.names : state.zoo_returns;
    const pos = new Map(main.dates.map((d, i) => [d, i]));
    const ls = zoo.dates.map(d => main.returns[m].LS[pos.get(d)]);
    const x = [monthBefore(zoo.dates[0]), ...zoo.dates];
    const traces = [{ x, y: wealth(ls), name: m, mode: "lines", line: { color: "red", width: 4 }, legendrank: 1 }];
    const idx = new Map(zoo.names.map((n, i) => [n, i]));
    for (const n of chosen) {
      traces.push({ x, y: wealth(zoo.returns[idx.get(n)].map(v => isNum(v) ? v / 100 : NaN)), name: n, mode: "lines",
        line: { color: "grey", width: 2 }, opacity: 0.1, legendrank: 2 + traces.length });
    }
    // draw the model last so it sits on top; legendrank keeps it first in the legend
    plot(div, [...traces.slice(1), traces[0]], { height: 600, yaxis: { type: "log", title: "" }, xaxis: { title: "", dtick: "M12", tickformat: "%Y" },
      margin: { l: 60, r: 20, t: 20, b: 40 } });
  });
  p.append(multiselect("Select zoo returns to display:", ["Select All", ...zoo.names], "zoo_returns", ["Select All"], () => chart.refresh()), chart);
}

function detailsMovingAverage(p, main, m) {
  p.append(el("h2", {}, "Moving Average Returns"));
  const chart = region(div => {
    const w = state.ma_window;
    const roll = r => {
      const v = r.map(x => (isNum(x) ? x : 0) * 100);
      const out = new Array(v.length).fill(null), half = Math.floor(w / 2);
      for (let i = 0; i < v.length; i++) {
        const a = i - half, b = a + w - 1;
        if (a < 0 || b >= v.length) continue;
        let s = 0; for (let j = a; j <= b; j++) s += v[j];
        out[i] = s / w;
      }
      return out;
    };
    const traces = binTraces(main, m, roll, { x: main.dates });
    plot(div, traces, { title: `${w}-Month Moving Average Returns for ${m} Model`, height: 550,
      yaxis: { title: "Monthly Return (%)", zeroline: false }, xaxis: { title: "Date" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "black", width: 0.8, dash: "dash" } }],
      legend: { x: 0.99, xanchor: "right", y: 0.01, yanchor: "bottom", bgcolor: "rgba(255,255,255,0.7)" }, hovermode: "x unified" });
  });
  p.append(slider("Moving average window (months):", 3, 24, 1, "ma_window", 12, () => chart.refresh()), md("#md-details-ma"), chart);
}

function detailsRegression(p, main, m) {
  p.append(el("h2", {}, "Regression Analysis (a la Table 1)"), md("#md-details-reg"));
  const t = main.table1[m];
  p.append(alphaTable(t, t.cols.map(c => c === "binLS" ? "Long-Short" : c)), md("#md-regression-notes"));
}

function detailsStatistics(p, main, m) {
  p.append(el("h2", {}, "Model Statistics"));
  const s = main.sharpe[m];
  const left = el("div", {}, el("h3", {}, "Sharpe Ratios"),
    dataTable(["Sharpe Ratio"], Object.entries(s).map(([k, v]) => ({ label: k, values: { "Sharpe Ratio": v } })), { defaultFormat: fmt.f(4) }));
  const t = main.turnoverSummary[m];
  const right = el("div", {}, el("h3", {}, "Turnover Analysis"),
    dataTable(["% Buy", "% Hold"], t.bins.map((b, i) => ({ label: b, values: { "% Buy": t.buy[i], "% Hold": t.hold[i] } })),
      { defaultFormat: fmt.pct(1), indexName: "bin" }));
  p.append(columns(left, right));
}

function detailsTurnover(p, main, m) {
  p.append(el("h2", {}, "Turnover Patterns"));
  const h = main.hold[m];
  const traces = Object.keys(h).map((b, i) => ({ x: h[b].dates, y: h[b].hold, name: b, mode: "lines",
    line: { width: 1.5, color: ["#f1c3a6", "#e0846a", "#b4485d", "#6f2c5e", "#2d1e3e"][i] } }));
  plot(p, traces, { title: "Percentage of Stocks Held Over Time", height: 420, yaxis: { title: "Hold Percentage", range: [0, 1], tickformat: ".0%" },
    xaxis: { title: "Date" }, legend: { title: { text: "Portfolio" } } });
}

async function yearPicker(container, m, key, onYear) {
  const years = (await load("model_years"))[m] || [];
  if (!years.length) {
    container.append(callout("warning", "No saved pipelines found for this model. Re-run run_linear_backtest.py / run_nn_backtest.py / run_hgb_backtest.py."));
    return null;
  }
  const k = `${key}_${m}`;
  remember(k, years[years.length - 1]);
  const content = region(async div => {
    const info = await load(`models/${encodeURIComponent(m)}/${state[k]}`);
    if (info.error) { div.append(callout("error", info.error)); return; }
    onYear(div, info);
  });
  container.append(selectbox("Test year (the model was trained only on earlier years)", years, k, state[k], () => content.refresh()), content);
}

async function detailsPipeline(p, m) {
  p.append(el("h2", {}, "Visualization of Model Pipeline"));
  await yearPicker(p, m, "pipeline_year", (div, info) => {
    if (info.pipelineHtml) {
      const frame = el("iframe", { class: "pipeline", title: "Model pipeline" });
      frame.srcdoc = info.pipelineHtml;
      div.append(frame);
    }
    if (info.columnGroups?.length) {
      div.append(el("h2", {}, "Which columns go through which transformers?"));
      div.append(columns(...info.columnGroups.slice(0, 2).map((g, i) => el("div", {},
        el("p", {}, el("strong", {}, `Columns in the ${g.name} transformer`)),
        frameTable({ columns: [i === 0 ? "Scroll to see all" : "Scroll to see applicable columns"], rows: g.columns.map(c => [c]) },
          { maxHeight: 400, index: false })))));
    }
  });
}

async function detailsImportance(p, m) {
  p.append(el("h2", {}, "Which variables are important in this model?"));
  let current = null;
  const chart = region(div => {
    if (!current) return;
    const imp = current.importance;
    if (!imp) { div.append(callout("info", "This model does not expose feature importances in a way the dashboard can display.")); return; }
    const n = state.top_n;
    const names = imp.names.slice(0, n), values = imp.values.slice(0, n);
    div.append(el("h3", {}, imp.subheader));
    plot(div, [{ type: "bar", orientation: "h", x: values, y: names, marker: { color: "#636efa" },
      hovertemplate: `${imp.x_label}=%{x}<br>Feature=%{y}<extra></extra>` }],
    { title: imp.title, height: Math.max(400, names.length * 25), xaxis: { title: imp.x_label },
      yaxis: { title: "Feature", categoryorder: "total ascending", automargin: true },
      hoverlabel: { bgcolor: "white", font: { size: 12 } }, margin: { l: 20, r: 20, t: 40, b: 40 } });
    if (imp.caption) div.append(caption(imp.caption));
  });
  p.append(slider("Top N features", 5, 50, 5, "top_n", 20, () => chart.refresh()));
  await yearPicker(p, m, "importance_year", (div, info) => { current = info; div.append(chart); chart.refresh(); });
}

// ----- Trading the Model in Practice -----------------------------------------------------------------------
async function practicePage(page) {
  page.append(el("h1", {}, "Trading the Model in Practice"));
  page.append(tabs("practice_tab", [
    ["Background", p => p.append(md("#md-practice-bg"))],
    ["Trading in Your 401(k)", p => practice401k(p)],
    ["Tax-Loss Harvesting", p => taxLossHarvesting(p)],
  ]));
}

function edgeTests(r, dates, F) {
  const g = c => F.get(c, dates);
  const RF = g("RF"), MKT = g("mkt_excess"), SMB = g("SMB"), HML = g("HML"), RMW = g("RMW"), CMA = g("CMA"), MOM = g("Mom");
  const n = r.length;
  const lags = Math.floor(4 * Math.pow(n / 100, 2 / 9));
  const excess = r.map((v, i) => 100 * v - RF[i]);
  const minusMkt = r.map((v, i) => 100 * v - (MKT[i] + RF[i]));
  const fit = (y, cols, hac) => {
    const ok = y.map((v, i) => isNum(v) && cols.every(c => isNum(c[i])));
    const yy = y.filter((_, i) => ok[i]);
    const X = y.map((_, i) => [1, ...cols.map(c => c[i])]).filter((_, i) => ok[i]);
    return ols(yy, X, hac ? lags : null);
  };
  const rows = [];
  const plain = fit(minusMkt, [], false), nw = fit(minusMkt, [], true);
  rows.push({ label: "Return minus market (ordinary t)", values: { "Per year (%)": 12 * plain.params[0], "t-statistic": plain.tvalues[0] } });
  rows.push({ label: "Return minus market (Newey-West t)", values: { "Per year (%)": 12 * nw.params[0], "t-statistic": nw.tvalues[0] } });
  for (const [name, cols] of [["CAPM alpha", [MKT]], ["Fama-French 3-factor alpha", [MKT, SMB, HML]],
    ["Fama-French 6-factor alpha", [MKT, SMB, HML, RMW, CMA, MOM]]]) {
    const f = fit(excess, cols, true);
    rows.push({ label: name, values: { "Per year (%)": 12 * f.params[0], "t-statistic": f.tvalues[0], Beta: f.params[1] } });
  }
  return rows;
}

async function practice401k(p) {
  p.append(el("h2", {}, "How Each Way of Trading the Model Would Have Done"));
  const [practice, main] = await Promise.all([load("practice"), load("main")]);
  const F = factorLookup(main);
  const models = practice.models;
  const realisticName = main.realisticApproach;
  const body = region(div => {
    const model = state.practice_model, start = state.practice_start;
    const hs = state.practice_half_spread / 10000, fee = state.practice_borrow_fee / 100;
    const rows = [], nets = {};
    let d;
    for (const approach of practice.approaches) {
      d = since(practice.data[model][approach], start, ["gross", "net_spread_estimate", "turnover", "trades"]);
      const longOnly = approach.startsWith("Long only");
      const net = approach === "Long-short, all stocks" ? d.net_spread_estimate
        : d.gross.map((g, i) => g - 2 * hs * d.turnover[i] - (longOnly ? 0 : fee / 12));
      nets[approach] = { r: net, dates: d.dates };
      const trades = 12 * mean(d.trades.filter((_, i) => d.dates[i] !== practice.firstMonth));
      const rf = F.rf(d.dates), mkt = F.market(d.dates);
      for (const [label, r] of [[approach, d.gross], [`${approach}, after costs`, net]]) {
        const s = returnStats(r, longOnly ? rf : null);
        rows.push({ label, values: { "Mean (%/yr)": s["Mean (%/yr)"], "Sharpe ratio": s["Sharpe ratio"], "Max drawdown (%)": s["Max drawdown (%)"],
          "Above market (%/yr)": longOnly ? 1200 * mean(sub(r, mkt)) : NaN, "Trades/yr": trades } });
      }
    }
    const s = returnStats(F.market(d.dates), F.rf(d.dates));
    rows.push({ label: "Index fund (stock market)", values: { "Mean (%/yr)": s["Mean (%/yr)"], "Sharpe ratio": s["Sharpe ratio"],
      "Max drawdown (%)": s["Max drawdown (%)"], "Above market (%/yr)": 0, "Trades/yr": NaN } });
    const realistic = `${realisticName}, after costs`;
    div.append(dataTable(["Mean (%/yr)", "Sharpe ratio", "Max drawdown (%)", "Above market (%/yr)", "Trades/yr"], rows,
      { formats: { "Sharpe ratio": fmt.f(2), "Trades/yr": fmt.int }, rowClass: l => l === realistic ? "highlight" : null }));
    div.append(caption("The sample ends in December 2019. The hold-until-rank-60 rule was picked once, not tuned, but with " +
      "only 20 stocks, a few lucky picks can move these results a lot."));
    if (nets[realisticName]) {
      div.append(el("h3", {}, "Is the realistic version's edge over the market reliable?"), md("#md-practice-edge", { realistic }));
      const n = nets[realisticName];
      div.append(dataTable(["Per year (%)", "t-statistic", "Beta"], edgeTests(n.r, n.dates, F), { formats: { "t-statistic": fmt.f(2), Beta: fmt.f(2) } }));
    }
  });
  remember("practice_model", models.includes("MLP-rank-256-128-64") ? "MLP-rank-256-128-64" : models[0]);
  remember("practice_start", 2005);
  remember("practice_half_spread", 5); remember("practice_borrow_fee", 0.25);
  const refresh = () => body.refresh();
  p.append(columns(selectbox("Model:", models, "practice_model", null, refresh),
    radio("Sample starts in:", [2000, 2005, 2010], "practice_start", 2005, refresh)));
  p.append(md("#md-practice-401k"));
  p.append(columns(numberInput("Half-spread for liquid stocks, each way (basis points)", 0, 100, 1, "practice_half_spread", 5, refresh),
    numberInput("Borrowing fee on short legs (% per year)", 0, 10, 0.25, "practice_borrow_fee", 0.25, refresh)));
  p.append(body);
}

function taxLossHarvesting(p) {
  p.append(el("h2", {}, "Tax-Loss Harvesting: A Small Example"), md("#md-tlh-intro"));
  const money = v => "$" + Math.round(v).toLocaleString("en-US");
  const body = region(div => {
    const st = state.tlh_st / 100, lt = state.tlh_lt / 100;
    const plans = [
      ["Plan 1: Follow the model", "Sell A and C.", 4000 + 1000, 0],
      ["Plan 2: Also harvest B's loss", "Sell A and C, and sell B and put the $7,000 into a similar stock.", 4000 + 1000 - 3000, 0],
      ["Plan 3: Harvest, and wait a month on C", "Sell A, harvest B, and sell C in February, after a year.", 4000 - 3000, 1000],
    ];
    const cols = ["What you do", "Net short-term gain", "Long-term gain", "Tax"];
    div.append(dataTable(cols, plans.map(([name, what, s, l]) => ({ label: name, values: {
      "What you do": what, "Net short-term gain": money(s), "Long-term gain": money(l), Tax: money(st * Math.max(s, 0) + lt * l) } })),
    { defaultFormat: v => v, indexName: "Plan" }));
    const base = st * 5000;
    div.append(md("#md-tlh-after", { save2: Math.round(base - st * 2000).toLocaleString("en-US"),
      save3: Math.round(st * 1000 - lt * 1000).toLocaleString("en-US") }));
  });
  remember("tlh_st", 37); remember("tlh_lt", 20);
  p.append(columns(numberInput("Short-term rate (your income tax rate, %)", 0, 60, 1, "tlh_st", 37, () => body.refresh()),
    numberInput("Long-term rate (%)", 0, 40, 1, "tlh_lt", 20, () => body.refresh())), body);
}

// ----- Background & Methods --------------------------------------------------------------------------------
async function backgroundPage(page) {
  page.append(el("h1", {}, "Financial Asset Pricing Evaluation: A Step-by-Step Guide"));
  const bg = await load("background");
  page.append(tabs("background_tab", [
    ["(1) Overview", p => p.append(el("h3", {}, "Overview"), md("#md-bg-overview"))],
    ["(2) Returns Data", p => p.append(el("h3", {}, "Returns Data"), md("#md-bg-returns"), frameTable(bg.toyRets, { maxHeight: 400 }))],
    ["(3) Signals Data", p => p.append(el("h3", {}, "Data on Signals: Predictive Features"), md("#md-bg-signals"), frameTable(bg.toySignals, { maxHeight: 400 }))],
    ["(4) Combine Return and Signals Data", p => p.append(el("h3", {}, "Combined Data: Ready for Prediction"), md("#md-bg-combined"), frameTable(bg.combined, { maxHeight: 400 }))],
    ["(5) Training Timeline", p => backgroundTimeline(p, bg)],
    ["(6) Portfolio Construction", p => p.append(columns(
      el("div", {}, el("h3", {}, "Portfolio Construction: From Predictions to Portfolio Positions"), md("#md-bg-portfolio")),
      el("div", {}, el("h3", {}, "Predictions sorted into portfolios"), frameTable(bg.portfolios, { maxHeight: 400, index: false }),
        el("h3", {}, "Portfolio returns"), frameTable(bg.portRets, { index: false }))))],
    ["(7) Full Code Example", p => backgroundNotebook(p, bg)],
  ]));
}

function backgroundTimeline(p, bg) {
  p.append(el("h3", {}, "Walk-Forward Validation: How Models Learn"), md("#md-bg-timeline"));
  const trainWindow = 5;
  const table = region(div => {
    const nf = state.tl_firms;
    const t = bg.timeline[nf];
    const mi = t.columns.indexOf("yyyymm");
    const months = [...new Set(t.rows.map(r => r[mi]))];
    const predIdx = Math.min(state.tl_pred, months.length - 1);
    const train = new Set(months.slice(0, predIdx)), pred = months[predIdx];
    const lo = trainWindow * nf, hi = predIdx * nf + nf;
    const rows = t.rows.map((r, i) => [...r, i >= lo && i < hi ? t.fake[i] : null]);
    div.append(frameTable({ columns: [...t.columns, "prediction"], rows }, {
      rowStyle: r => train.has(r[mi]) ? { background: "lightgreen" } : r[mi] === pred ? { background: "lightsalmon" } : null }));
  });
  const nMonths = new Set(bg.timeline[1].rows.map(r => r[1])).size;
  const left = el("div", {},
    slider("Number of Firms to Display", 1, 4, 1, "tl_firms", 1, () => table.refresh()),
    slider("Select Prediction Month", trainWindow, nMonths - 1, 1, "tl_pred", trainWindow, () => table.refresh()),
    table);
  p.append(columns(left, md("#md-bg-timeline-code")));
}

function backgroundNotebook(p, bg) {
  p.append(el("h2", {}, "Full Code Example"), md("#md-bg-notebook"));
  const titles = bg.notebook.filter(c => c.type === "markdown" && c.source.startsWith("## ")).map(c => c.source.split("\n")[0].slice(3).trim());
  p.append(el("h2", {}, "Outline of Notebook"), el("ol", {}, titles.map(t => el("li", {}, el("a", { href: "#" + slug(t) }, t)))), el("hr"));
  for (const c of bg.notebook) {
    if (c.type === "markdown") p.append(md(c.source));
    else {
      const code = el("code", { class: "language-python" }, c.source);
      if (window.hljs) hljs.highlightElement(code);
      p.append(el("pre", { class: "code" }, code));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Router and shell
// ---------------------------------------------------------------------------------------------------------
function currentPage() {
  const m = location.hash.match(/^#\/(\w+)/);
  return PAGES.find(p => p[0] === (m && m[1])) || PAGES[0];
}

function route() {
  const [key, , render] = currentPage();
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.page === key));
  const page = document.getElementById("page");
  page.replaceChildren();
  window.scrollTo(0, 0);
  Promise.resolve(render(page)).catch(err => {
    console.error(err);
    page.append(callout("error", `Could not load this page: ${err.message}`));
  });
  document.body.classList.remove("nav-open");
}

function start() {
  const nav = document.getElementById("nav");
  for (const [key, label] of PAGES) {
    nav.append(el("a", { href: `#/${key}`, "data-page": key }, el("span", { class: "dot" }), label));
  }
  document.getElementById("sidebar-toggle").addEventListener("click", () => document.body.classList.toggle("nav-open"));
  // In-page anchors (#some-heading) scroll instead of changing the page
  document.addEventListener("click", e => {
    const a = e.target.closest("a[href^='#']");
    if (!a || a.getAttribute("href").startsWith("#/")) return;
    e.preventDefault();
    const target = document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  window.addEventListener("hashchange", route);
  route();
}

document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", start) : start();

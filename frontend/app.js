(function () {
  "use strict";

  const API_BASE = (window.API_BASE_URL || "").replace(/\/$/, "");

  /* ── DOM refs ─────────────────────────────────────────────────── */
  const $ = (id) => document.getElementById(id);
  const statusDot       = $("statusDot");
  const statusText      = $("statusText");
  const backendState    = $("backendState");
  const backendMeta     = $("backendMeta");
  const loadedStat      = $("loadedStat");
  const loadedMeta      = $("loadedMeta");
  const selectedStat    = $("selectedStat");
  const selectedMeta    = $("selectedMeta");
  const bestAdeStat     = $("bestAdeStat");
  const bestAdeMeta     = $("bestAdeMeta");
  const modelListEl     = $("modelList");
  const modelCountEl    = $("modelCount");
  const modelSearchEl   = $("modelSearch");
  const loadedOnlyEl    = $("loadedOnlyToggle");
  const selectLoadedBtn = $("selectLoadedBtn");
  const selectTopBtn    = $("selectTopBtn");
  const clearSelBtn     = $("clearSelectionBtn");
  const modelHintEl     = $("modelHint");
  const lbBody          = $("leaderboardBody");
  const selSummary      = $("selectionSummary");
  const runBtn          = $("runDemoBtn");
  const clearBtn        = $("clearBtn");
  const datasetSel      = $("datasetSelect");
  const metricsEl       = $("liveMetricsText");
  const progressEl      = $("progressText");
  const csvUploadEl     = $("csvUpload");
  const trajLoadingEl   = $("trajLoading");
  const trajPlaceholder = $("trajPlaceholder");
  const errPlaceholder  = $("errPlaceholder");
  const trajPlotEl      = $("trajectoryPlot");
  const errPlotEl       = $("errorPlot");
  const animProgressWrap = $("animProgressWrap");
  const animProgressFill = $("animProgressFill");
  const liveMetricsCards = $("liveMetricsCards");
  const resultsEmpty     = $("resultsEmpty");
  const resultTrajWrapper = $("resultTrajWrapper");
  const resultErrWrapper  = $("resultErrWrapper");
  const resultTrajPlot   = $("resultTrajPlot");
  const resultErrPlot    = $("resultErrPlot");
  const errorSummaryEl   = $("errorSummaryContent");
  const dlTrajCsv        = $("dlTrajCsv");
  const dlErrorCsv       = $("dlErrorCsv");
  const dlTrajPng        = $("dlTrajPng");
  const dlErrPng         = $("dlErrPng");
  const themeToggleBtn   = $("themeToggle");
  const animSpeedEl      = $("animSpeed");
  const toastContainerEl = $("toastContainer");
  const validationModal  = $("validationModal");
  const validationIcon   = $("validationIcon");
  const validationTitle  = $("validationTitle");
  const validationBody   = $("validationBody");
  const validationClose  = $("validationClose");
  const validationOkBtn  = $("validationOkBtn");
  /* ── State ────────────────────────────────────────────────────── */
  let models = [];
  let lbRows = [];
  const byName   = new Map();
  const selected = new Set();
  const ui = { query: "", loadedOnly: false };
  let lastSimResponse = null;
  let animationId     = null;

  /* ── Theme System ─────────────────────────────────────────────── */
  const THEME_KEY = "ins-gps-theme";

  const MOON_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
  const SUN_SVG  = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;

  function getSystemTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function getStoredTheme() { return localStorage.getItem(THEME_KEY); }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
    if (themeToggleBtn) themeToggleBtn.innerHTML = theme === "dark" ? SUN_SVG : MOON_SVG;
    // Re-layout plots with new theme colours
    setTimeout(relayoutPlotsForTheme, 50);
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(cur === "dark" ? "light" : "dark");
  }

  /* ── CSS variable helper ──────────────────────────────────────── */
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ── Dynamic Plotly theme relayout ───────────────────────────── */
  function relayoutPlotsForTheme() {
    if (typeof Plotly === "undefined") return;
    const update = {
      plot_bgcolor:            cssVar("--plot-bg"),
      paper_bgcolor:           cssVar("--plot-bg"),
      "xaxis.gridcolor":       cssVar("--plot-grid"),
      "yaxis.gridcolor":       cssVar("--plot-grid"),
      "xaxis.linecolor":       cssVar("--plot-axis"),
      "yaxis.linecolor":       cssVar("--plot-axis"),
      "xaxis.tickcolor":       cssVar("--plot-axis"),
      "yaxis.tickcolor":       cssVar("--plot-axis"),
      "xaxis.zerolinecolor":   cssVar("--plot-axis"),
      "yaxis.zerolinecolor":   cssVar("--plot-axis"),
      "xaxis.tickfont.color":  cssVar("--plot-tick"),
      "yaxis.tickfont.color":  cssVar("--plot-tick"),
      "xaxis.title.font.color":cssVar("--plot-tick"),
      "yaxis.title.font.color":cssVar("--plot-tick"),
      "title.font.color":      cssVar("--plot-title"),
      "legend.bgcolor":        cssVar("--plot-legend-bg"),
      "legend.bordercolor":    cssVar("--border"),
      "legend.font.color":     cssVar("--plot-title"),
    };
    [trajPlotEl, errPlotEl, resultTrajPlot, resultErrPlot].forEach(el => {
      try { if (el && el.data && el.data.length > 0) Plotly.relayout(el, update); }
      catch (_) {}
    });
  }

  /* ── Theme-aware Ground Truth colours ────────────────────────── */
  function isDark() { return document.documentElement.getAttribute("data-theme") !== "light"; }
  function gtColor()     { return isDark() ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.5)"; }
  function gtGlowColor() { return isDark() ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)"; }

  /* ── Animation duration from slider ──────────────────────────── */
  function getAnimDuration() {
    const val = animSpeedEl ? parseInt(animSpeedEl.value, 10) : 4;
    // val 1 = 8s (slowest) … val 10 = 0.8s (fastest)
    return 8000 - (val - 1) * 800;
  }

  /* ── Toast notifications ──────────────────────────────────────── */
  const TOAST_ICONS = {
    success: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
    error:   `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
    info:    `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
  };

  function showToast(message, type = "info", duration = 3600) {
    if (!toastContainerEl) return;
    const div = document.createElement("div");
    div.className = `toast ${type}`;
    div.innerHTML = `<span class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span><span>${message}</span>`;
    toastContainerEl.appendChild(div);
    setTimeout(() => {
      div.classList.add("out");
      setTimeout(() => div.remove(), 320);
    }, duration);
  }

  /* ── Keyboard shortcuts ───────────────────────────────────────── */
  document.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.code === "Space" && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      if (!runBtn.disabled) runSimulation();
    }
    if (e.code === "Escape") clearAll();
    if ((e.key === "d" || e.key === "D") && !e.ctrlKey && !e.metaKey) toggleTheme();
  });

  /* ── Tab System ───────────────────────────────────────────────── */
  function initTabs(containerEl) {
    const btns = containerEl.querySelectorAll(".tab-btn");
    btns.forEach(btn => {
      btn.addEventListener("click", () => {
        btns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const card = containerEl.closest(".card") || containerEl.parentElement;
        card.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        const target = card.querySelector(`#panel-${btn.dataset.tab}`);
        if (target) target.classList.add("active");
      });
    });
  }

  function switchTab(containerEl, tabId) {
    const btn = containerEl.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.click();
  }

  /* ── Network ──────────────────────────────────────────────────── */
  async function api(path, opts) {
    const r = await fetch(`${API_BASE}${path}`, opts);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  /* ── Status helpers ───────────────────────────────────────────── */
  function setStatus(state, text) {
    statusDot.className = `status-dot ${state}`;
    statusText.textContent = text;
    backendState.textContent = state === "error" ? "offline" : state === "online" ? "online" : "…";
    backendMeta.textContent  = text;
  }

  function fmtNum(n, d = 3) { return n != null ? Number(n).toFixed(d) : "—"; }

  function visible() {
    const q = ui.query.trim().toLowerCase();
    return models.filter(m => {
      if (ui.loadedOnly && !m.loaded) return false;
      return !q || m.model_name.toLowerCase().includes(q);
    });
  }

  /* Animate a stat value with a pop class */
  function animateValue(el, newText) {
    el.textContent = newText;
    el.classList.remove("value-pop");
    void el.offsetWidth;
    el.classList.add("value-pop");
  }

  function showPlot(plotEl, phEl) {
    phEl.classList.add("hidden");
    plotEl.style.display = "block";
  }
  function showPlaceholder(plotEl, phEl) {
    plotEl.style.display = "none";
    phEl.classList.remove("hidden");
    if (typeof Plotly !== "undefined") { try { Plotly.purge(plotEl); } catch (_) {} }
  }

  /* ── Stats strip ──────────────────────────────────────────────── */
  function syncStats() {
    const loaded = models.filter(m => m.loaded).length;
    animateValue(loadedStat, loaded);
    loadedMeta.textContent  = loaded === models.length ? "All artifacts present." : `${models.length - loaded} missing.`;
    animateValue(selectedStat, selected.size);
    selectedMeta.textContent = selected.size ? "Ready for simulation." : "Select models to compare.";
    const best = lbRows[0];
    animateValue(bestAdeStat, best?.["ADE (m)"] != null ? `${fmtNum(best["ADE (m)"])} m` : "—");
    bestAdeMeta.textContent = best?.model_name || "From leaderboard.";
    modelCountEl.textContent = `${loaded}/${models.length}`;
    if (selected.size > 0) runBtn.classList.add("pulse");
    else                   runBtn.classList.remove("pulse");
  }

  /* ── Selection summary ────────────────────────────────────────── */
  function renderSelSummary() {
    if (!selected.size) {
      selSummary.innerHTML = `<div class="selection-empty">Select models from the left panel to compare their trajectories.</div>`;
      return;
    }
    selSummary.innerHTML = Array.from(selected).map(name => {
      const m   = byName.get(name); if (!m) return "";
      const row = lbRows.find(r => r.model_name === name) || {};
      const ade = row["ADE (m)"] != null ? `${fmtNum(row["ADE (m)"])} m` : "—";
      const params = m.param_count != null ? m.param_count.toLocaleString() : "—";
      return `<div class="selection-card">
        <div><strong style="color:${m.color}">${name}</strong> <span>${ade}</span></div>
        <div><span>${params} params</span></div>
      </div>`;
    }).join("");
  }

  function applySelection(names) {
    selected.clear();
    names.forEach(n => { if (byName.get(n)?.loaded) selected.add(n); });
    renderChips();
    renderSelSummary();
    syncStats();
  }

  /* ── Model chips ──────────────────────────────────────────────── */
  function renderChips() {
    modelListEl.innerHTML = "";
    const vis = visible();
    if (!vis.length) {
      modelListEl.innerHTML = `<div class="empty-state" style="padding:18px 0">No models match.</div>`;
      modelHintEl.textContent = "Adjust your filters.";
      return;
    }
    modelHintEl.textContent = `${vis.length} model(s) · click to select`;

    vis.forEach((m, idx) => {
      const el = document.createElement("div");
      el.className = "chip" +
        (m.loaded ? "" : " disabled") +
        (selected.has(m.model_name) ? " active" : "");
      el.style.setProperty("--swatch", m.color);
      el.style.animationDelay = `${idx * 28}ms`;

      const ade    = m.metrics?.["ADE (m)"] != null ? `${fmtNum(m.metrics["ADE (m)"])}m` : (m.loaded ? "—" : "n/a");
      const params = m.param_count != null ? `${(m.param_count / 1000).toFixed(1)}K` : "";

      el.innerHTML = `
        <span class="chip-dot"></span>
        <span class="chip-name">${m.model_name}</span>
        <span class="chip-info">
          <span class="chip-ade">${ade}</span>
          <span class="chip-params">${params}</span>
        </span>`;

      if (m.loaded) {
        el.addEventListener("click", () => {
          selected.has(m.model_name) ? selected.delete(m.model_name) : selected.add(m.model_name);
          renderChips();
          renderSelSummary();
          syncStats();
        });
      }
      modelListEl.appendChild(el);
    });
  }

  /* ── Leaderboard ──────────────────────────────────────────────── */
  function renderLB(rows) {
    lbRows = rows;
    syncStats();
    if (!rows.length) {
      lbBody.innerHTML = `<tr><td colspan="6"><div class="empty-state">No artifacts loaded.</div></td></tr>`;
      return;
    }
    const max = Math.max(...rows.map(r => r["ADE (m)"] ?? 0), 0.001);
    lbBody.innerHTML = rows.map((r, i) => {
      const ade  = r["ADE (m)"];
      const pct  = ade != null ? Math.max(4, (ade / max) * 100) : 0;
      const m    = byName.get(r.model_name) || {};
      const prms = m.param_count != null ? m.param_count.toLocaleString() : "—";
      const tt   = m.train_time_s != null ? `${fmtNum(m.train_time_s, 1)}s` : "—";
      return `<tr>
        <td class="rank">${i + 1}</td>
        <td><div class="model-cell"><span style="color:${r.color}">${r.model_name}</span><small>${m.loaded ? "Loaded" : "Missing"}</small></div></td>
        <td class="ade-cell">${fmtNum(ade)}</td>
        <td class="meta-cell">${prms}</td>
        <td class="meta-cell">${tt}</td>
        <td class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${r.color}"></div></div></td>
      </tr>`;
    }).join("");
    renderSelSummary();
  }

  /* ── Dark Plotly layouts (CSS-driven) ────────────────────────── */
  function basePlotLayout(overrides = {}) {
    return {
      plot_bgcolor:  cssVar("--plot-bg"),
      paper_bgcolor: cssVar("--plot-bg"),
      font: { family: "Inter, sans-serif", color: cssVar("--plot-title"), size: 12 },
      xaxis: {
        gridcolor:     cssVar("--plot-grid"),
        linecolor:     cssVar("--plot-axis"),
        tickcolor:     cssVar("--plot-axis"),
        zerolinecolor: cssVar("--plot-axis"),
        tickfont:  { size: 10, color: cssVar("--plot-tick") },
        titlefont: { size: 11, color: cssVar("--plot-tick") },
        mirror: true, showline: true, ticks: "outside",
        ...overrides.xaxis,
      },
      yaxis: {
        gridcolor:     cssVar("--plot-grid"),
        linecolor:     cssVar("--plot-axis"),
        tickcolor:     cssVar("--plot-axis"),
        zerolinecolor: cssVar("--plot-axis"),
        tickfont:  { size: 10, color: cssVar("--plot-tick") },
        titlefont: { size: 11, color: cssVar("--plot-tick") },
        mirror: true, showline: true, ticks: "outside",
        ...overrides.yaxis,
      },
      legend: {
        x: 1, y: 1, xanchor: "right", yanchor: "top",
        bgcolor:     cssVar("--plot-legend-bg"),
        bordercolor: cssVar("--border"),
        borderwidth: 1,
        font: { size: 10, color: cssVar("--plot-title") },
      },
      hovermode: "closest",
      margin: { t: 42, r: 22, b: 50, l: 56 },
      ...overrides,
    };
  }

  function trajLayout(title, rangePE, rangePN) {
    return basePlotLayout({
      title: { text: title, font: { size: 13, color: cssVar("--plot-title") } },
      xaxis: { title: { text: "PE (m)", standoff: 6 }, range: rangePE },
      yaxis: { title: { text: "PN (m)", standoff: 6 }, range: rangePN },
      height: 510,
    });
  }

  function errLayout(rangeX, rangeY) {
    return basePlotLayout({
      title: { text: "Position Error Over Time", font: { size: 12, color: cssVar("--plot-title") } },
      xaxis: { title: { text: "Window Index" }, range: rangeX },
      yaxis: { title: { text: "Error (m)" }, rangemode: "tozero", range: rangeY },
      height: 250,
      margin: { t: 34, r: 22, b: 44, l: 56 },
    });
  }

  function resultTrajLayout(title, rangePE, rangePN) {
    const lay = trajLayout(title, rangePE, rangePN);
    lay.height = 460;
    return lay;
  }

  function resultErrLayout(rangeX, rangeY) {
    const lay = errLayout(rangeX, rangeY);
    lay.height = 230;
    return lay;
  }

  /* ── Colour helpers ───────────────────────────────────────────── */
  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  /* ── Simulation ───────────────────────────────────────────────── */
  async function runSimulation() {
    if (!selected.size) { showToast("Select at least one model first.", "error"); return; }
    const file = datasetSel.value;
    if (!file) { showToast("No dataset selected.", "error"); return; }
    if (typeof Plotly === "undefined") { showToast("Plotly.js failed to load — check internet connection.", "error"); return; }

    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }

    runBtn.disabled = true;
    runBtn.classList.remove("pulse");
    progressEl.textContent = "processing…";
    trajLoadingEl.classList.add("active");
    switchTab($("centerTabs"), "liveView");

    try {
      const names = Array.from(selected);
      const response = await api("/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_names: names, dataset: file }),
      });

      lastSimResponse = response;
      trajLoadingEl.classList.remove("active");

      /* ── Global bounds ─────────────────────── */
      let minPE = Math.min(...response.gt_PE), maxPE = Math.max(...response.gt_PE);
      let minPN = Math.min(...response.gt_PN), maxPN = Math.max(...response.gt_PN);
      let maxErr = 0.1;

      response.results.forEach(r => {
        minPE = Math.min(minPE, ...r.pred_PE); maxPE = Math.max(maxPE, ...r.pred_PE);
        minPN = Math.min(minPN, ...r.pred_PN); maxPN = Math.max(maxPN, ...r.pred_PN);
        maxErr = Math.max(maxErr, ...r.errors);
      });

      const padPE = (maxPE - minPE) * 0.06 || 10;
      const padPN = (maxPN - minPN) * 0.06 || 10;
      const rangePE   = [minPE - padPE, maxPE + padPE];
      const rangePN   = [minPN - padPN, maxPN + padPN];
      const numWindows = response.gt_PN.length;
      const rangeErrX  = [0, numWindows];
      const rangeErrY  = [0, maxErr * 1.12];

      /* ── Trajectory traces (glow + main + head dot) ─────────── */
      const traces = [];

      // GT glow
      traces.push({ x: [response.gt_PE[0]], y: [response.gt_PN[0]], mode: "lines",
        name: "Ground Truth", showlegend: false,
        line: { color: gtGlowColor(), width: 10 }, hoverinfo: "skip" });
      // GT main
      traces.push({ x: [response.gt_PE[0]], y: [response.gt_PN[0]], mode: "lines",
        name: "Ground Truth",
        line: { color: gtColor(), width: 2.5 },
        hovertemplate: "GT<br>PE: %{x:.1f}m<br>PN: %{y:.1f}m<extra></extra>" });

      response.results.forEach(r => {
        traces.push({ x: [r.pred_PE[0]], y: [r.pred_PN[0]], mode: "lines",
          name: r.model_name, showlegend: false,
          line: { color: hexToRgba(r.color, 0.15), width: 8 }, hoverinfo: "skip" });
        traces.push({ x: [r.pred_PE[0]], y: [r.pred_PN[0]], mode: "lines",
          name: r.model_name + (response.results.length === 1 ? " (est.)" : ""),
          line: { color: r.color, width: 2.5, dash: response.results.length === 1 ? "dash" : "solid" },
          hovertemplate: r.model_name + "<br>PE: %{x:.1f}m<br>PN: %{y:.1f}m<extra></extra>" });
      });

      // Start marker
      traces.push({ x: [response.gt_PE[0]], y: [response.gt_PN[0]], mode: "markers",
        name: "Start",
        marker: { color: cssVar("--accent"), size: 10, symbol: "circle",
                  line: { width: 2, color: cssVar("--accent-glow") } },
        hovertemplate: "Start<br>PE: %{x:.1f}m<br>PN: %{y:.1f}m<extra></extra>" });

      // Drawing head dot (last trace, hidden after animation)
      traces.push({ x: [response.gt_PE[0]], y: [response.gt_PN[0]], mode: "markers",
        name: "Head", showlegend: false,
        marker: { color: cssVar("--accent"), size: 9, symbol: "circle",
                  line: { width: 2, color: "#fff" } },
        hoverinfo: "skip" });

      const headTraceIdx = traces.length - 1;

      const titleText = names.length === 1
        ? `Trajectory: ${names[0]} vs Ground Truth`
        : `Trajectory: All Models vs Ground Truth`;

      showPlot(trajPlotEl, trajPlaceholder);
      Plotly.newPlot(trajPlotEl, traces, trajLayout(titleText, rangePE, rangePN),
        { responsive: true, displayModeBar: false });

      /* ── Error traces (glow + main) ──────────────────────────── */
      const errTraces = [];
      response.results.forEach(r => {
        errTraces.push({ x: [0], y: [r.errors[0]], mode: "lines", name: r.model_name,
          showlegend: false, line: { color: hexToRgba(r.color, 0.12), width: 6 }, hoverinfo: "skip" });
        errTraces.push({ x: [0], y: [r.errors[0]], mode: "lines", name: r.model_name,
          line: { color: r.color, width: 1.5 },
          hovertemplate: r.model_name + "<br>Window: %{x}<br>Error: %{y:.2f}m<extra></extra>" });
      });

      showPlot(errPlotEl, errPlaceholder);
      Plotly.newPlot(errPlotEl, errTraces, errLayout(rangeErrX, rangeErrY),
        { responsive: true, displayModeBar: false });

      /* ── Smooth animation loop ───────────────────────────────── */
      const ANIM_MS = getAnimDuration();
      const startTime = performance.now();
      let prevDrawn = 1;

      animProgressWrap.classList.add("active");
      animProgressFill.style.width = "0%";

      function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

      function animLoop(now) {
        const elapsed      = now - startTime;
        const rawProgress  = Math.min(elapsed / ANIM_MS, 1);
        const easedProg    = easeOutCubic(rawProgress);
        const targetIdx    = Math.max(1, Math.floor(easedProg * numWindows));

        animProgressFill.style.width = `${Math.round(rawProgress * 100)}%`;

        if (targetIdx <= prevDrawn && rawProgress < 1) {
          animationId = requestAnimationFrame(animLoop);
          return;
        }

        const drawFrom = prevDrawn;
        const drawTo   = Math.min(targetIdx, numWindows);

        if (drawTo > drawFrom) {
          // Trajectory extension
          const xTraj = [], yTraj = [], tIdxs = [];
          const gx = response.gt_PE.slice(drawFrom, drawTo);
          const gy = response.gt_PN.slice(drawFrom, drawTo);
          xTraj.push(gx, gx); yTraj.push(gy, gy); tIdxs.push(0, 1);

          response.results.forEach((r, i) => {
            const sx = r.pred_PE.slice(drawFrom, drawTo);
            const sy = r.pred_PN.slice(drawFrom, drawTo);
            xTraj.push(sx, sx); yTraj.push(sy, sy);
            tIdxs.push(2 + i * 2, 3 + i * 2);
          });
          Plotly.extendTraces(trajPlotEl, { x: xTraj, y: yTraj }, tIdxs);

          // Update head dot to current GT position
          const curPE = response.gt_PE[drawTo - 1] ?? response.gt_PE[0];
          const curPN = response.gt_PN[drawTo - 1] ?? response.gt_PN[0];
          Plotly.restyle(trajPlotEl, { x: [[curPE]], y: [[curPN]] }, [headTraceIdx]);

          // Error extension
          const xErr = [], yErr = [], eIdxs = [];
          const idxArr = Array.from({ length: drawTo - drawFrom }, (_, k) => drawFrom + k);
          response.results.forEach((r, i) => {
            const es = r.errors.slice(drawFrom, drawTo);
            xErr.push(idxArr, idxArr); yErr.push(es, es); eIdxs.push(i * 2, i * 2 + 1);
          });
          Plotly.extendTraces(errPlotEl, { x: xErr, y: yErr }, eIdxs);
        }

        prevDrawn = drawTo;

        if (rawProgress < 1) {
          animationId = requestAnimationFrame(animLoop);
        } else {
          animationId = null;
          animProgressFill.style.width = "100%";
          // Hide head dot
          Plotly.restyle(trajPlotEl, { visible: false }, [headTraceIdx]);
          setTimeout(() => animProgressWrap.classList.remove("active"), 700);
          onSimulationComplete(response, names, rangePE, rangePN, rangeErrX, rangeErrY);
        }
      }

      animationId = requestAnimationFrame(animLoop);

      /* Live metrics display */
      metricsEl.textContent = response.results
        .map(r => `${r.model_name}: ADE ${r.ade.toFixed(2)}m · FDE ${r.fde.toFixed(2)}m`)
        .join("  |  ");
      progressEl.textContent = `${numWindows} windows`;
      renderLiveMetrics(response);

    } catch (err) {
      trajLoadingEl.classList.remove("active");
      animProgressWrap.classList.remove("active");
      setStatus("error", err.message);
      showToast(err.message, "error");
      console.error("Simulation error:", err);
    } finally {
      runBtn.disabled = false;
    }
  }

  /* ── Post-simulation ──────────────────────────────────────────── */
  function onSimulationComplete(response, names, rangePE, rangePN, rangeErrX, rangeErrY) {
    setStatus("online", "Simulation complete");
    buildResultPlots(response, names, rangePE, rangePN, rangeErrX, rangeErrY);
    enableDownloads(response);
    renderErrorSummary(response);
    showToast("Simulation complete!", "success");
    setTimeout(() => switchTab($("centerTabs"), "results"), 900);
  }

  /* ── Static result plots ──────────────────────────────────────── */
  function buildResultPlots(response, names, rangePE, rangePN, rangeErrX, rangeErrY) {
    resultsEmpty.style.display = "none";
    resultTrajWrapper.style.display = "block";
    resultErrWrapper.style.display  = "block";

    const trajTraces = [];
    trajTraces.push({ x: response.gt_PE, y: response.gt_PN, mode: "lines",
      name: "Ground Truth",
      line: { color: gtColor(), width: 2.5 },
      hovertemplate: "GT<br>PE: %{x:.1f}m<br>PN: %{y:.1f}m<extra></extra>" });

    response.results.forEach(r => {
      trajTraces.push({ x: r.pred_PE, y: r.pred_PN, mode: "lines",
        name: r.model_name,
        line: { color: r.color, width: 2.5, dash: response.results.length === 1 ? "dash" : "solid" },
        hovertemplate: r.model_name + "<br>PE: %{x:.1f}m<br>PN: %{y:.1f}m<extra></extra>" });
    });

    if (response.gt_PN.length > 0) {
      trajTraces.push({ x: [response.gt_PE[0]], y: [response.gt_PN[0]], mode: "markers",
        name: "Start", marker: { color: cssVar("--accent"), size: 10, symbol: "circle",
          line: { width: 2, color: cssVar("--accent-glow") } } });
      const last = response.gt_PE.length - 1;
      trajTraces.push({ x: [response.gt_PE[last]], y: [response.gt_PN[last]], mode: "markers",
        name: "End", marker: { color: cssVar("--danger"), size: 10, symbol: "square",
          line: { width: 2, color: cssVar("--danger-dim") } } });
    }

    const titleText = names.length === 1
      ? `Trajectory: ${names[0]} vs Ground Truth`
      : `Trajectory: All Models vs Ground Truth`;

    Plotly.newPlot(resultTrajPlot, trajTraces,
      resultTrajLayout(titleText, rangePE, rangePN),
      { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ["lasso2d","select2d"], displaylogo: false });

    const numWindows = response.gt_PN.length;
    const xFull      = Array.from({ length: numWindows }, (_, i) => i);
    const errTraces  = response.results.map(r => ({
      x: xFull, y: r.errors, mode: "lines", name: r.model_name,
      line: { color: r.color, width: 1.5 },
      hovertemplate: r.model_name + "<br>Window: %{x}<br>Error: %{y:.2f}m<extra></extra>",
    }));

    Plotly.newPlot(resultErrPlot, errTraces,
      resultErrLayout(rangeErrX, rangeErrY),
      { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ["lasso2d","select2d"], displaylogo: false });
  }

  /* ── Live metrics (Overview tab) ──────────────────────────────── */
  function renderLiveMetrics(response) {
    if (!response?.results?.length) { liveMetricsCards.innerHTML = ""; return; }
    liveMetricsCards.innerHTML =
      `<div style="font-size:11px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Simulation Results</div>` +
      response.results.map(r => `
        <div class="live-metric-card">
          <div class="lmc-name" style="color:${r.color}">${r.model_name}</div>
          <div class="lmc-row">
            <span>ADE: ${r.ade.toFixed(3)}m</span>
            <span>FDE: ${r.fde.toFixed(3)}m</span>
            <span>${r.inference_ms.toFixed(0)}ms</span>
          </div>
        </div>`).join("");
  }

  /* ── Error summary (Downloads tab) ───────────────────────────── */
  function renderErrorSummary(response) {
    if (!response?.results?.length) {
      errorSummaryEl.innerHTML = `<div class="dl-empty">Run a simulation to see error metrics.</div>`;
      return;
    }
    const rows = response.results.map(r => {
      const maxE = Math.max(...r.errors);
      const minE = Math.min(...r.errors);
      return `<tr>
        <td class="model-name-cell" style="color:${r.color}">${r.model_name}</td>
        <td>${r.ade.toFixed(3)}</td>
        <td>${r.fde.toFixed(3)}</td>
        <td>${maxE.toFixed(2)}</td>
        <td>${minE.toFixed(3)}</td>
        <td>${r.inference_ms.toFixed(0)}</td>
      </tr>`;
    }).join("");
    errorSummaryEl.innerHTML = `
      <table class="error-summary-table">
        <thead><tr><th>Model</th><th>ADE</th><th>FDE</th><th>Max</th><th>Min</th><th>ms</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  /* ── Downloads ────────────────────────────────────────────────── */
  function enableDownloads() {
    [dlTrajCsv, dlErrorCsv, dlTrajPng, dlErrPng].forEach(b => b.disabled = false);
  }

  function downloadCSV(filename, content) {
    const url  = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8;" }));
    const link = Object.assign(document.createElement("a"), { href: url, download: filename });
    link.click();
    URL.revokeObjectURL(url);
  }

  function exportTrajectoryCSV() {
    if (!lastSimResponse) return;
    const r = lastSimResponse;
    const n = r.gt_PN.length;
    let hdr = "window_index,gt_PE,gt_PN";
    r.results.forEach(res => hdr += `,${res.model_name}_PE,${res.model_name}_PN`);
    let csv = hdr + "\n";
    for (let i = 0; i < n; i++) {
      let row = `${i},${r.gt_PE[i].toFixed(6)},${r.gt_PN[i].toFixed(6)}`;
      r.results.forEach(res => row += `,${res.pred_PE[i].toFixed(6)},${res.pred_PN[i].toFixed(6)}`);
      csv += row + "\n";
    }
    downloadCSV("trajectory_data.csv", csv);
    showToast("Trajectory CSV downloaded.", "success");
  }

  function exportErrorCSV() {
    if (!lastSimResponse) return;
    const r = lastSimResponse;
    const n = r.gt_PN.length;
    let hdr = "window_index";
    r.results.forEach(res => hdr += `,${res.model_name}_error`);
    let csv = hdr + "\n";
    for (let i = 0; i < n; i++) {
      let row = `${i}`;
      r.results.forEach(res => row += `,${res.errors[i].toFixed(6)}`);
      csv += row + "\n";
    }
    downloadCSV("error_data.csv", csv);
    showToast("Error CSV downloaded.", "success");
  }

  function exportPlotPNG(plotEl, filename) {
    if (typeof Plotly === "undefined") return;
    Plotly.toImage(plotEl, { format: "png", width: 1920, height: 1080, scale: 2 })
      .then(url => { Object.assign(document.createElement("a"), { href: url, download: filename }).click(); });
    showToast("Exporting PNG…", "info");
  }

  /* ── Clear ────────────────────────────────────────────────────── */
  function clearAll() {
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
    showPlaceholder(trajPlotEl, trajPlaceholder);
    showPlaceholder(errPlotEl, errPlaceholder);
    animProgressWrap.classList.remove("active");
    metricsEl.textContent = "ADE: — | FDE: —";
    progressEl.textContent = "";
    lastSimResponse = null;
    liveMetricsCards.innerHTML = "";
    resultsEmpty.style.display = "";
    resultTrajWrapper.style.display = "none";
    resultErrWrapper.style.display  = "none";
    try { Plotly.purge(resultTrajPlot); } catch (_) {}
    try { Plotly.purge(resultErrPlot); } catch (_) {}
    [dlTrajCsv, dlErrorCsv, dlTrajPng, dlErrPng].forEach(b => b.disabled = true);
    errorSummaryEl.innerHTML = `<div class="dl-empty">Run a simulation to see error metrics.</div>`;
    switchTab($("centerTabs"), "liveView");
  }

  /* ── File upload with validation ──────────────────────────────── */
  const MAX_CLIENT_SIZE = 50 * 1024 * 1024; // 50 MB

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Client-side pre-checks
    if (!file.name.endsWith('.csv')) {
      showToast("Only .csv files are supported.", "error");
      e.target.value = "";
      return;
    }
    if (file.size > MAX_CLIENT_SIZE) {
      showToast(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is 50 MB.`, "error");
      e.target.value = "";
      return;
    }
    if (file.size === 0) {
      showToast("File is empty.", "error");
      e.target.value = "";
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    try {
      setStatus("", "uploading & validating…");
      showToast("Uploading & validating dataset…", "info");
      const r = await fetch(`${API_BASE}/upload_dataset`, { method: "POST", body: formData });
      if (!r.ok) {
        const errData = await r.json().catch(() => null);
        throw new Error(errData?.detail || `Upload failed: ${r.status}`);
      }
      const report = await r.json();

      // Refresh dataset list
      const dsets = await api("/datasets");
      datasetSel.innerHTML = dsets.datasets.map(d =>
        `<option value="${d}" ${d === report.filename ? "selected" : ""}>${d}</option>`).join("");

      // Show validation report modal
      showValidationReport(report);

      if (report.valid) {
        setStatus("online", `Uploaded ${report.filename}`);
      } else {
        setStatus("error", `${report.filename}: validation issues`);
      }
    } catch (err) {
      setStatus("error", err.message);
      showToast(err.message, "error");
      console.error(err);
    }
    e.target.value = "";
  }

  /* ── Validation report modal ─────────────────────────────────── */
  function showValidationReport(report) {
    // Icon
    const hasErrors = report.issues?.some(i => i.severity === "error");
    const hasWarnings = report.issues?.some(i => i.severity === "warning");
    let iconClass, iconSvg;

    if (!report.valid || hasErrors) {
      iconClass = "error";
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    } else if (hasWarnings) {
      iconClass = "warning";
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    } else {
      iconClass = "success";
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
    }

    validationIcon.className = `modal-icon ${iconClass}`;
    validationIcon.innerHTML = iconSvg;
    validationTitle.textContent = report.valid
      ? `✓ ${report.filename} — Ready for Simulation`
      : `⚠ ${report.filename} — Validation Issues`;

    // Build body HTML
    let html = '';

    // Summary stats grid
    html += `<div class="vr-summary">`;
    html += `<div class="vr-stat"><div class="vr-stat-label">Total Rows</div><div class="vr-stat-value">${(report.total_rows ?? 0).toLocaleString()}</div></div>`;
    html += `<div class="vr-stat"><div class="vr-stat-label">Valid Rows</div><div class="vr-stat-value ${report.valid_rows >= 50 ? 'ok' : 'bad'}">${(report.valid_rows ?? 0).toLocaleString()}</div></div>`;
    html += `<div class="vr-stat"><div class="vr-stat-label">Ground Truth</div><div class="vr-stat-value ${report.gt_format !== 'none' ? 'ok' : 'bad'}">${report.gt_format || 'none'}</div></div>`;
    html += `<div class="vr-stat"><div class="vr-stat-label">Usable Windows</div><div class="vr-stat-value ${(report.usable_windows || 0) > 0 ? 'ok' : 'bad'}">${(report.usable_windows ?? 0).toLocaleString()}</div></div>`;
    html += `</div>`;

    // Column check
    if (report.sensor_cols_present?.length || report.sensor_cols_missing?.length) {
      html += `<div class="vr-section">`;
      html += `<div class="vr-section-title"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>Required Sensor Columns (${report.sensor_cols_present?.length || 0}/${(report.sensor_cols_present?.length || 0) + (report.sensor_cols_missing?.length || 0)})</div>`;
      html += `<div class="vr-cols">`;
      (report.sensor_cols_present || []).forEach(c => {
        // Shorten the column name for display
        const short = c.replace(/ \(.*\)/, '');
        html += `<span class="vr-col-tag present" title="${c}">✓ ${short}</span>`;
      });
      (report.sensor_cols_missing || []).forEach(c => {
        const short = c.replace(/ \(.*\)/, '');
        html += `<span class="vr-col-tag missing" title="${c}">✗ ${short}</span>`;
      });
      html += `</div></div>`;
    }

    // Issues
    if (report.issues?.length) {
      html += `<div class="vr-section">`;
      html += `<div class="vr-section-title"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>Issues (${report.issues.length})</div>`;
      report.issues.forEach(issue => {
        const sevIcon = issue.severity === 'error'
          ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
          : issue.severity === 'warning'
          ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
          : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
        html += `<div class="vr-issue ${issue.severity}"><span class="vr-issue-icon">${sevIcon}</span><span>${issue.message}</span></div>`;
      });
      html += `</div>`;
    }

    // Cleaning actions
    if (report.cleaning_applied?.length) {
      html += `<div class="vr-section">`;
      html += `<div class="vr-section-title"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>Auto-Cleaning Applied</div>`;
      report.cleaning_applied.forEach(action => {
        html += `<div class="vr-cleaning"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>${action}</div>`;
      });
      html += `</div>`;
    }

    // Column stats table (show only if there are stats)
    const statsKeys = Object.keys(report.column_stats || {});
    if (statsKeys.length) {
      html += `<div class="vr-section">`;
      html += `<div class="vr-section-title"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>Column Statistics</div>`;
      html += `<table class="vr-col-stats"><thead><tr><th>Column</th><th>Min</th><th>Max</th><th>Mean</th><th>Bad</th><th>Outliers</th></tr></thead><tbody>`;
      statsKeys.forEach(col => {
        const s = report.column_stats[col];
        // Shorten column name
        const short = col.length > 25 ? col.replace(/ \(.*\)/, '') : col;
        html += `<tr>
          <td style="font-family:var(--font);font-size:10px;">${short}</td>
          <td>${s.min != null ? s.min.toFixed(2) : '—'}</td>
          <td>${s.max != null ? s.max.toFixed(2) : '—'}</td>
          <td>${s.mean != null ? s.mean.toFixed(2) : '—'}</td>
          <td style="color:${s.non_numeric > 0 ? 'var(--danger)' : 'var(--text-tertiary)'}">${s.non_numeric ?? 0}</td>
          <td style="color:${s.outliers > 0 ? 'var(--warning)' : 'var(--text-tertiary)'}">${s.outliers ?? 0}</td>
        </tr>`;
      });
      html += `</tbody></table></div>`;
    }

    // If all green and no issues
    if (report.valid && !report.issues?.length) {
      html += `<div class="vr-issue info"><span class="vr-issue-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span><span>All checks passed! Dataset is ready for simulation.</span></div>`;
    }

    validationBody.innerHTML = html;

    // Show modal
    validationModal.style.display = "flex";

    // Also show appropriate toast
    if (report.valid) {
      showToast(`${report.filename} validated — ${report.usable_windows || 0} windows ready`, "success");
    } else {
      showToast(`${report.filename} has validation issues — check report`, "error");
    }
  }

  function closeValidationModal() {
    validationModal.style.display = "none";
  }

  /* ── Init ─────────────────────────────────────────────────────── */
  async function init() {
    // Theme
    const savedTheme = getStoredTheme() || getSystemTheme();
    setTheme(savedTheme);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", e => {
      if (!getStoredTheme()) setTheme(e.matches ? "dark" : "light");
    });
    if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);

    // Tabs
    initTabs($("centerTabs"));
    initTabs($("rightTabs"));

    if (!API_BASE) { setStatus("error", "API_BASE_URL not configured"); return; }
    try {
      setStatus("", "connecting…");
      const [mods, lb, dsets] = await Promise.all([
        api("/models"), api("/leaderboard"), api("/datasets"),
      ]);
      models = mods;
      byName.clear();
      models.forEach(m => byName.set(m.model_name, m));
      renderChips();
      renderLB(lb.leaderboard);
      datasetSel.innerHTML = dsets.datasets.length
        ? dsets.datasets.map(d => `<option value="${d}">${d}</option>`).join("")
        : `<option value="">No datasets found</option>`;
      const loaded = models.filter(m => m.loaded).length;
      setStatus(loaded ? "online" : "error", loaded ? `${loaded}/${models.length} models ready` : "No models loaded");
      if (loaded) showToast(`Backend connected — ${loaded} models ready`, "success");
      renderSelSummary();
    } catch (err) {
      setStatus("error", "Backend unreachable");
      showToast("Backend unreachable. Is the server running?", "error");
      lbBody.innerHTML = `<tr><td colspan="6"><div class="empty-state">${err.message}</div></td></tr>`;
      console.error(err);
    }
  }

  /* ── Event bindings ───────────────────────────────────────────── */
  modelSearchEl.addEventListener("input",  () => { ui.query = modelSearchEl.value; renderChips(); });
  loadedOnlyEl.addEventListener("change",  () => { ui.loadedOnly = loadedOnlyEl.checked; renderChips(); });
  selectLoadedBtn.addEventListener("click", () => applySelection(models.filter(m => m.loaded).map(m => m.model_name)));
  selectTopBtn.addEventListener("click",    () => applySelection(lbRows.filter(r => byName.get(r.model_name)?.loaded).slice(0, 3).map(r => r.model_name)));
  clearSelBtn.addEventListener("click",     () => applySelection([]));
  runBtn.addEventListener("click",   runSimulation);
  clearBtn.addEventListener("click", clearAll);
  csvUploadEl.addEventListener("change", handleUpload);
  validationClose.addEventListener("click", closeValidationModal);
  validationOkBtn.addEventListener("click", closeValidationModal);
  validationModal.addEventListener("click", (e) => { if (e.target === validationModal) closeValidationModal(); });
  dlTrajCsv.addEventListener("click",  exportTrajectoryCSV);
  dlErrorCsv.addEventListener("click", exportErrorCSV);
  dlTrajPng.addEventListener("click",  () => exportPlotPNG(resultTrajPlot, "trajectory_plot.png"));
  dlErrPng.addEventListener("click",   () => exportPlotPNG(resultErrPlot,  "error_plot.png"));

  init();
})();

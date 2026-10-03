/* DietGate dashboard: polls /admin/* every 2s and renders panels. */
"use strict";

const $ = (id) => document.getElementById(id);
let adminKey = localStorage.getItem("dg-admin-key") || "admin-dev-key";
let controlKey = localStorage.getItem("dg-control-key") || "";
let curveChart = null;
let latencyChart = null;

$("adminKey").value = adminKey;
$("saveKey").onclick = () => {
  adminKey = $("adminKey").value.trim();
  localStorage.setItem("dg-admin-key", adminKey);
};
$("controlKey").value = controlKey;
$("saveControlKey").onclick = () => {
  controlKey = $("controlKey").value.trim();
  localStorage.setItem("dg-control-key", controlKey);
  applyControlAvailability();
};

function applyControlAvailability() {
  const enabled = controlKey.length > 0;
  document.querySelectorAll(".ctl").forEach((el) => (el.disabled = !enabled));
  $("sim-status").textContent = enabled
    ? $("sim-status").textContent
    : "disabled - enter the control key to use the controls";
}

async function api(path, options) {
  const resp = await fetch(path, {
    ...options,
    headers: { "X-Admin-Key": adminKey, "Content-Type": "application/json" },
  });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

async function controlApi(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "X-Control-Key": controlKey, "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err?.error?.message || `${path} -> ${resp.status}`);
  }
  return resp.json();
}

function fmtUsd(v) {
  if (v === null || v === undefined) return "-";
  v = Number(v);
  if (v !== 0 && Math.abs(v) < 0.01) return "$" + v.toFixed(6);
  return "$" + v.toFixed(4);
}
function fmtMs(v) {
  return v === null || v === undefined ? "-" : Number(v).toFixed(2) + " ms";
}

async function refreshCards() {
  const s = await api("/admin/summary?window=24h");
  $("c-spend").textContent = fmtUsd(s.spend_usd);
  $("c-baseline").textContent = fmtUsd(s.routed_baseline_spend_usd);
  $("c-savings").textContent =
    s.savings_pct === null
      ? "no routed traffic yet"
      : `savings ${s.savings_pct}% (model:auto, ${s.routed_requests} req)`;
  $("c-cps").textContent =
    s.success_rate ? fmtUsd(s.routed_spend_usd / Math.max(1, s.labeled_tasks)) : "-";
  $("c-labeled").textContent = `${s.labeled_tasks} labeled tasks`;
  $("c-cache").textContent =
    s.cache_hit_rate === null ? "-" : (s.cache_hit_rate * 100).toFixed(1) + "%";
  $("c-requests").textContent = `${s.requests} requests`;
  $("c-p50").textContent = fmtMs(s.overhead_p50_ms);
  $("c-p99").textContent = `p99 ${fmtMs(s.overhead_p99_ms)}`;
  $("c-success").textContent =
    s.success_rate === null ? "-" : (s.success_rate * 100).toFixed(1) + "%";
  $("c-window").textContent = `${s.labeled_tasks} labeled / ${s.requests} requests`;
}

async function refreshCurve() {
  const data = await api("/admin/learning_curve?bucket=10s&window=24h");
  const labels = data.series.map((p) =>
    new Date(p.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  );
  const actual = data.series.map((p) => p.cost_per_success_usd);
  const baseline = data.series.map((p) => p.baseline_per_success_usd);
  const explore = data.series.map((p) => (p.explore_share === null ? null : p.explore_share * 100));
  if (!curveChart) {
    const ctx = $("curve").getContext("2d");
    curveChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "DietGate $/success", data: actual, borderColor: "#4fd1a5", tension: 0.25, pointRadius: 0 },
          { label: "baseline $/success (always the biggest model)", data: baseline, borderColor: "#5b9dff", borderDash: [6, 4], tension: 0.1, pointRadius: 0 },
          { label: "explore share (right axis, %)", data: explore, borderColor: "#f2b950", yAxisID: "y2", tension: 0.2, pointRadius: 0, borderWidth: 1.5 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false, // fill the 300px-high box instead of a fixed 2:1 aspect ratio
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: {
            type: "linear",
            beginAtZero: true,
            title: { display: true, text: "$ per successful task" },
            ticks: { callback: (v) => "$" + Number(v).toFixed(4) },
          },
          y2: {
            position: "right",
            min: 0,
            max: 100,
            grid: { drawOnChartArea: false },
            title: { display: true, text: "explore share" },
            ticks: { callback: (v) => v + "%" },
          },
        },
      },
    });
    return;
  }
  curveChart.data.labels = labels;
  curveChart.data.datasets[0].data = actual;
  curveChart.data.datasets[1].data = baseline;
  curveChart.data.datasets[2].data = explore;
  curveChart.update("none");
}

async function refreshMatrix() {
  const data = await api("/admin/routing_matrix?window=24h");
  const tbody = $("matrix").querySelector("tbody");
  tbody.innerHTML = "";
  for (const row of data.matrix) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.task_type}</td><td>${row.model}</td>` +
      `<td class="num">${(row.share * 100).toFixed(1)}%</td>` +
      `<td class="num">${row.success_rate === null ? "-" : (row.success_rate * 100).toFixed(1) + "%"}</td>` +
      `<td class="num">${fmtUsd(row.avg_cost_usd)}</td>`;
    tbody.appendChild(tr);
  }
}

async function refreshLatency() {
  const data = await api("/admin/latency?window=24h");
  const labels = ["p50", "p90", "p99"];
  const series = [
    labels.map((l) => data.overhead_ms[l]),
    labels.map((l) => data.ttft_ms[l]),
    labels.map((l) => data.total_ms[l]),
  ];
  if (!latencyChart) {
    const ctx = $("latency").getContext("2d");
    latencyChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "gateway overhead ms", data: series[0], backgroundColor: "#4fd1a5" },
          { label: "TTFT ms (mock)", data: series[1], backgroundColor: "#5b9dff" },
          { label: "total ms (mock)", data: series[2], backgroundColor: "#324056" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          // log scale: a ~0.2 ms gateway overhead is invisible next to hundreds of ms of (mock) model latency
          y: { type: "logarithmic", min: 0.01, title: { display: true, text: "ms (log scale)" } },
        },
      },
    });
    return;
  }
  series.forEach((d, i) => (latencyChart.data.datasets[i].data = d));
  latencyChart.update("none");
}

async function refreshRecent() {
  const data = await api("/admin/requests?limit=12");
  const tbody = $("recent").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of data.requests) {
    const time = new Date(r.ts * 1000).toLocaleTimeString();
    const flags = [
      r.cache_hit ? "cache" : null,
      r.shortcut ? `shortcut:${r.shortcut}` : null,
      r.explore ? "explore" : null,
      r.attempts > 1 ? `attempts:${r.attempts}` : null,
      r.status !== 200 ? `status:${r.status}` : null,
    ].filter(Boolean).join(", ");
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${time}</td><td>${r.task_type}</td><td>${r.chosen_model || "-"}</td>` +
      `<td>${r.policy || "-"}</td><td>${(r.decision_reason || "").slice(0, 80)}</td>` +
      `<td class="num">${fmtUsd(r.cost_usd)}</td><td class="num">${fmtMs(r.overhead_ms)}</td>` +
      `<td>${flags}</td>`;
    tbody.appendChild(tr);
  }
}

async function refreshLearningCost() {
  const data = await api("/admin/learning_cost?window=24h");
  $("learning-cost").innerHTML =
    `exploration spend: <b>${fmtUsd(data.exploration_spend_usd)}</b> &nbsp;|&nbsp; ` +
    `shadow evaluations: <b>${data.shadow_evals}</b> (${fmtUsd(data.shadow_eval_spend_usd)})`;
}

async function refreshSim() {
  try {
    const s = await api("/admin/sim/status");
    $("sim-status").textContent = s.running
      ? `running: ${s.scenario} (${s.done}/${s.total} tasks)`
      : s.result
        ? `finished: ${s.result.tasks} tasks, savings shown above`
        : "idle";
  } catch {
    $("sim-status").textContent = "controls unavailable";
  }
}

async function tick() {
  try {
    await Promise.all([
      refreshCards(), refreshCurve(), refreshMatrix(),
      refreshLatency(), refreshRecent(), refreshLearningCost(), refreshSim(),
    ]);
    $("connState").textContent = "live";
    $("connState").className = "conn ok";
    $("sim-badge").classList.remove("hidden");
  } catch (err) {
    $("connState").textContent = "admin key? " + err.message;
    $("connState").className = "conn bad";
  }
}

$("sim-start").onclick = () =>
  controlApi("/admin/sim/start", { scenario: $("scenario").value }).then(tick).catch((e) => alert(e.message));
$("sim-stop").onclick = () => controlApi("/admin/sim/stop").then(tick).catch((e) => alert(e.message));
document.querySelectorAll("[data-policy]").forEach((btn) => {
  btn.onclick = () =>
    controlApi("/admin/policy", { policy: btn.dataset.policy }).then(tick).catch((e) => alert(e.message));
});
$("reset").onclick = () => {
  if (confirm("Delete all demo data and reset stats?")) {
    controlApi("/admin/reset").then(tick).catch((e) => alert(e.message));
  }
};

tick();
applyControlAvailability();
setInterval(tick, 2000);

const COLORS = [
  "#1d4ed8", "#047857", "#7c3aed", "#b45309",
  "#be185d", "#0e7490", "#c2410c", "#475569", "#15803d",
];

let samples = [];
let meta = {};
let charts = [];

async function loadGzipJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`No se pudo cargar ${url}`);
  const buf = await res.arrayBuffer();
  const stream = new DecompressionStream("gzip");
  const out = await new Response(new Blob([buf]).stream().pipeThrough(stream)).arrayBuffer();
  return JSON.parse(new TextDecoder().decode(out));
}

function renderProbs(probs) {
  const ul = document.getElementById("probs");
  ul.innerHTML = "";
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const top = entries[0][0];
  entries.forEach(([name, p]) => {
    const li = document.createElement("li");
    if (name === top) li.classList.add("top");
    const pct = (p * 100).toFixed(1);
    li.innerHTML = `
      <div class="prob-row ${name === top ? "top" : ""}">
        <span>${name}</span><span>${pct}%</span>
      </div>
      <div class="bar"><span style="width:${pct}%"></span></div>
    `;
    ul.appendChild(li);
  });
}

function destroyCharts() {
  charts.forEach((c) => c.destroy());
  charts = [];
}

function renderCharts(signal, channels) {
  destroyCharts();
  const grid = document.getElementById("charts");
  grid.innerHTML = "";
  channels.forEach((ch, i) => {
    const cell = document.createElement("div");
    cell.className = "chart-cell";
    const canvas = document.createElement("canvas");
    cell.appendChild(canvas);
    grid.appendChild(cell);
    const data = signal.map((row) => row[i]);
    charts.push(
      new Chart(canvas, {
        type: "line",
        data: {
          labels: data.map((_, j) => j),
          datasets: [{
            data,
            borderColor: COLORS[i % COLORS.length],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: true, text: ch, font: { size: 10, weight: "600" } },
          },
          scales: {
            x: { display: false },
            y: { ticks: { maxTicksLimit: 4, font: { size: 8 } } },
          },
        },
      })
    );
  });
}

function showSample(idx) {
  const s = samples[idx];
  if (!s) return;
  document.getElementById("idx-label").textContent = idx;
  document.getElementById("idx-range").value = idx;
  document.getElementById("idx-num").value = idx;

  const banner = document.getElementById("banner");
  banner.className = "banner " + (s.match ? "ok" : "bad");
  banner.textContent = s.match
    ? "Clasificación correcta"
    : "Error de clasificación";

  document.getElementById("true-label").textContent = s.y_true;
  const predBox = document.getElementById("pred-box");
  predBox.className = "label-box " + (s.match ? "ok" : "bad");
  document.getElementById("pred-label").textContent = s.y_pred;

  renderProbs(s.probs);
  renderCharts(s.signal, meta.channels);
}

function bindControls() {
  const max = samples.length - 1;
  document.getElementById("max-idx").textContent = max;
  const range = document.getElementById("idx-range");
  const num = document.getElementById("idx-num");
  range.max = max;
  num.max = max;

  const update = (v) => {
    const i = Math.min(Math.max(0, parseInt(v, 10) || 0), max);
    showSample(i);
  };
  range.addEventListener("input", () => update(range.value));
  num.addEventListener("change", () => update(num.value));
}

async function init() {
  try {
    meta = await (await fetch("/data/meta.json")).json();
    samples = await loadGzipJson("/data/samples.json.gz");
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("app").classList.remove("hidden");
    bindControls();
    showSample(0);
  } catch (e) {
    document.getElementById("loading").textContent =
      "Error al cargar datos: " + e.message;
  }
}

init();

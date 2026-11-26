const LEVEL_COLORS = {
  low: "#22c55e",
  moderate: "#eab308",
  high: "#f97316",
  severe: "#ef4444",
};

const map = L.map("map").setView([7.9, -1.2], 7);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let markers = {};
let riskData = [];
let chart = null;
let selectedId = null;

function fetchJSON(url) {
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  });
}

function renderMarkers() {
  riskData.forEach((entry) => {
    if (!entry.peak) return;
    const { region, peak } = entry;
    const color = LEVEL_COLORS[peak.level] || "#8b9bb4";
    if (markers[region.id]) markers[region.id].remove();
    const marker = L.circleMarker([region.lat, region.lon], {
      radius: 12,
      color: "#0b1220",
      weight: 2,
      fillColor: color,
      fillOpacity: 0.9,
    }).addTo(map);
    marker.bindPopup(
      `<b>${region.name}</b><br/>Peak risk (7d): ${peak.level} ` +
        `(${Math.round(peak.probability * 100)}%)`
    );
    marker.on("click", () => selectRegion(region.id));
    markers[region.id] = marker;
  });
}

function renderList() {
  const container = document.getElementById("region-list");
  container.innerHTML = "";
  [...riskData]
    .sort((a, b) => (b.peak?.probability ?? -1) - (a.peak?.probability ?? -1))
    .forEach((entry) => {
      const div = document.createElement("div");
      div.className = "region-item";
      const color = entry.peak ? LEVEL_COLORS[entry.peak.level] : "#8b9bb4";
      const prob = entry.peak
        ? `${Math.round(entry.peak.probability * 100)}%`
        : "warming…";
      div.innerHTML = `<span class="dot" style="background:${color}"></span>
        <span class="region-name">${entry.region.name}</span>
        <span class="region-prob">${prob}</span>`;
      div.onclick = () => selectRegion(entry.region.id);
      container.appendChild(div);
    });
}

async function selectRegion(regionId) {
  selectedId = regionId;
  const detail = document.getElementById("region-detail");
  detail.innerHTML = "<p class='muted'>Loading forecast…</p>";

  try {
    const data = await fetchJSON(`/api/forecast/${regionId}?days=7`);
    const region = data.region;
    const daily = data.daily.slice(-7);
    if (!daily.length) throw new Error("no forecast yet");

    const peak = daily.reduce((a, b) => (b.probability > a.probability ? b : a));
    const color = LEVEL_COLORS[peak.level] || "#8b9bb4";
    detail.innerHTML = `
      <h2>${region.name} <span class="risk-badge" style="background:${color}">
        ${peak.level} · ${Math.round(peak.probability * 100)}%</span></h2>
      <p class="muted">Capital: ${region.capital} · Next 7 days ·
        ${peak.date}: peak flood risk</p>
      <canvas id="forecast-chart" height="130"></canvas>
      <table class="daily-table">
        <tr><th>Date</th><th>Rain (mm)</th><th>Discharge</th><th>Risk</th></tr>
        ${daily
          .map(
            (d) => `<tr>
              <td>${d.date}</td>
              <td>${d.precip}</td>
              <td>${d.discharge}</td>
              <td style="color:${LEVEL_COLORS[d.level]}">${d.level}</td>
            </tr>`
          )
          .join("")}
      </table>`;

    renderChart(daily);
  } catch (err) {
    detail.innerHTML = `<p class="muted">Forecast unavailable: ${err.message}</p>`;
  }
}

function renderChart(daily) {
  const ctx = document.getElementById("forecast-chart");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: daily.map((d) => d.date.slice(5)),
      datasets: [
        {
          label: "Flood probability",
          data: daily.map((d) => d.probability),
          backgroundColor: daily.map((d) => LEVEL_COLORS[d.level]),
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1, ticks: { callback: (v) => v * 100 + "%" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function applyRiskData(payload) {
  riskData = payload.regions;
  renderMarkers();
  renderList();
  document.getElementById("refresh-info").textContent =
    `Live forecast · ${riskData.length}/16 regions` +
    (payload.pending.length ? ` · warming ${payload.pending.length}…` : "");

  if (!selectedId && riskData.length) {
    const worst = [...riskData]
      .filter((r) => r.peak)
      .sort((a, b) => b.peak.probability - a.peak.probability)[0];
    if (worst) selectRegion(worst.region.id);
  }
  return payload.pending;
}

function pollRisk(triesLeft) {
  fetchJSON("/api/risk")
    .then((payload) => {
      const pending = applyRiskData(payload);
      if (pending.length && triesLeft > 0) {
        setTimeout(() => pollRisk(triesLeft - 1), 45000);
      }
    })
    .catch(() => {
      if (triesLeft > 0) setTimeout(() => pollRisk(triesLeft - 1), 45000);
      else
        document.getElementById("refresh-info").textContent =
          "Could not load risk data — is the API running?";
    });
}

pollRisk(12);

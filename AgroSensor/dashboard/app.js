// Detect if accessed via public URL or localhost
// Force public mode for agroaiapp.me domain regardless of what hostname reports
var isPublic = window.location.hostname !== '127.0.0.1' && window.location.hostname !== 'localhost';
if (window.location.hostname.includes('agroaiapp.me')) {
    isPublic = true;
}
// Also check if we're being served through tunnel (hostname might be misleading)
if (window.location.href.includes('sensor-dashboard.agroaiapp.me')) {
    isPublic = true;
}

var API = isPublic ? 'https://api.agroaiapp.me' : 'http://127.0.0.1:8000';
var API_PREFIX = '/api/sensor';
var isSimulationMode = false;
console.log('Hostname:', window.location.hostname, 'isPublic:', isPublic, 'API:', API);

// Simulated data for demonstration mode
function generateSimulatedData() {
    const baseTemp = 25 + Math.random() * 5;
    return {
        temperature: baseTemp,
        moisture: 45 + Math.random() * 15,
        ec: 1200 + Math.random() * 400,
        ph: 6.5 + Math.random() * 1.5,
        nitrogen: 80 + Math.random() * 40,
        phosphorus: 50 + Math.random() * 30,
        potassium: 150 + Math.random() * 100,
        timestamp: new Date().toISOString()
    };
}
let REFRESH_MS = 300000;
let historyChart = null;
let currentRange = '24h';
let currentCity = '';  // Active city for filtering
let citiesData = [];   // All cities from API (includes id, is_active)

// Parameter definitions: display ranges, optimal ranges, formatting
const PARAMS = {
    temperature: { min: 0,  max: 50,   optMin: 20, optMax: 35, fmt: v => v.toFixed(1),  color: '#ef4444' },
    moisture:    { min: 0,  max: 100,  optMin: 30, optMax: 70, fmt: v => v.toFixed(1),  color: '#3b82f6' },
    ec:          { min: 0,  max: 3000, optMin: 500, optMax: 2000, fmt: v => Math.round(v), color: '#f59e0b' },
    ph:          { min: 3,  max: 9,    optMin: 5.5, optMax: 7.5, fmt: v => v.toFixed(2), color: '#a855f7' },
    nitrogen:    { min: 0,  max: 300,  optMin: 50,  optMax: 200, fmt: v => Math.round(v), color: '#22c55e' },
    phosphorus:  { min: 0,  max: 250,  optMin: 30,  optMax: 150, fmt: v => Math.round(v), color: '#06b6d4' },
    potassium:   { min: 0,  max: 300,  optMin: 50,  optMax: 250, fmt: v => Math.round(v), color: '#f97316' },
};

// ─── Health & Status ───
async function fetchHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        if (!res.ok) throw new Error();
        // If we get here, API is working - not in simulation mode
        if (isSimulationMode) {
            isSimulationMode = false;
            console.log('API connection restored, switching to live mode');
        }
        const d = await res.json();

        const isMqtt = d.data_source === 'mqtt_cloud';

        // Sensor / Cloud label
        const sLabel = document.getElementById('sensor-label');
        sLabel.textContent = isMqtt ? '☁️ Cloud' : 'Sensor';

        // Sensor dot
        const sDot = document.getElementById('sensor-dot');
        const sVal = document.getElementById('sensor-status');
        sDot.className = 'status-dot ' + (d.sensor_connected ? 'online' : 'offline');
        sVal.textContent = d.sensor_connected ? 'Connected' : 'Disconnected';

        // MQTT status item (show only in cloud mode)
        const mqttItem = document.getElementById('mqtt-status-item');
        if (isMqtt && mqttItem) {
            mqttItem.style.display = '';
            const mDot = document.getElementById('mqtt-dot');
            const mVal = document.getElementById('mqtt-status');
            mDot.className = 'status-dot ' + (d.mqtt_connected ? 'online' : 'offline');
            mVal.textContent = d.mqtt_connected ? `Online (${d.mqtt_messages} msgs)` : 'Disconnected';
        }

        // DB dot
        const dDot = document.getElementById('db-dot');
        const dVal = document.getElementById('db-status');
        dDot.className = 'status-dot ' + (d.database_connected ? 'online' : 'offline');
        dVal.textContent = d.database_connected ? 'Connected' : 'Disconnected';

        // Uptime
        const ut = document.getElementById('uptime-val');
        const s = Math.floor(d.uptime_seconds);
        const h = Math.floor(s / 3600); const m = Math.floor((s % 3600) / 60);
        ut.textContent = h > 0 ? `${h}h ${m}m` : `${m}m ${s % 60}s`;

        // Last read
        if (d.last_reading_at) {
            document.getElementById('last-read-val').textContent =
                new Date(d.last_reading_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // Poll info footer
        if (d.polling_interval_seconds) {
            REFRESH_MS = d.polling_interval_seconds * 1000;
            const src = isMqtt ? 'MQTT Cloud' : 'Serial';
            document.getElementById('poll-info').textContent = `Source: ${src} | Poll: ${d.polling_interval_seconds}s`;
        }
    } catch (e) {
        document.getElementById('sensor-dot').className = 'status-dot offline';
        document.getElementById('sensor-status').textContent = isPublic ? 'Demo Mode' : 'Offline';
        document.getElementById('db-dot').className = 'status-dot offline';
        document.getElementById('db-status').textContent = isPublic ? 'Simulation' : 'Offline';
        // Mark as simulation mode for public users
        if (isPublic) {
            isSimulationMode = true;
            document.getElementById('sensor-label').textContent = '🎮 Demo';
        }
    }
}

// ─── Live Data ───
async function fetchLive() {
    try {
        const q = currentCity ? `?city=${encodeURIComponent(currentCity)}` : '';
        const url = `${API}${API_PREFIX}/live${q}`;
        console.log('Fetching live data from:', url);
        const res = await fetch(url);
        if (!res.ok) {
            console.error('Live data fetch failed:', res.status, res.statusText);
            // Switch to simulation mode for public access
            if (isPublic && !isSimulationMode) {
                console.log('Switching to simulation mode for public access');
                isSimulationMode = true;
            }
            if (isSimulationMode) {
                const simData = generateSimulatedData();
                updateCards(simData);
            }
            return;
        }
        const data = await res.json();
        console.log('Live data received:', data);
        updateCards(data);
    } catch (e) {
        console.error('fetchLive error:', e);
        // Use simulation data on error for public users
        if (isPublic) {
            isSimulationMode = true;
            const simData = generateSimulatedData();
            updateCards(simData);
        }
    }
}

function getStatus(key, val) {
    const p = PARAMS[key];
    if (val >= p.optMin && val <= p.optMax) return 'optimal';
    const lowWarn = p.optMin - (p.optMax - p.optMin) * 0.3;
    const highWarn = p.optMax + (p.optMax - p.optMin) * 0.3;
    if (val >= lowWarn && val <= highWarn) return 'warning';
    return 'critical';
}

function updateCards(data) {
    console.log('updateCards called with data:', data);
    for (const [key, cfg] of Object.entries(PARAMS)) {
        const val = data[key];
        console.log(`Processing ${key}:`, val);
        if (val == null) {
            console.warn(`Value for ${key} is null or undefined`);
            continue;
        }

        // Value
        const el = document.getElementById(`val-${key}`);
        if (el) {
            el.textContent = cfg.fmt(val);
            console.log(`Updated ${key} to:`, cfg.fmt(val));
        } else {
            console.warn(`Element not found: val-${key}`);
        }

        // Range bar + thumb
        const pct = Math.min(100, Math.max(0, ((val - cfg.min) / (cfg.max - cfg.min)) * 100));
        const bar = document.getElementById(`bar-${key}`);
        const thumb = document.getElementById(`thumb-${key}`);
        if (bar) {
            bar.style.width = pct + '%';
            const status = getStatus(key, val);
            bar.style.background = status === 'optimal' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444';
        }
        if (thumb) {
            thumb.style.left = pct + '%';
            const status = getStatus(key, val);
            thumb.style.borderColor = status === 'optimal' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444';
        }

        // Status dot
        const dot = document.getElementById(`status-${key}`);
        if (dot) dot.className = 'param-status ' + getStatus(key, val);
    }
}

// ─── History ───
async function fetchHistory() {
    try {
        // Calculate actual time-based start for the range
        const now = new Date();
        let start;
        let limit;
        if (currentRange === '7d') {
            start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
            limit = 2000;
        } else {
            start = new Date(now.getTime() - 24 * 60 * 60 * 1000);
            limit = 1000;
        }
        let url = `${API}${API_PREFIX}/history?limit=${limit}&start=${start.toISOString()}`;
        if (currentCity) url += `&city=${encodeURIComponent(currentCity)}`;
        console.log('Fetching history from:', url);
        const res = await fetch(url);
        if (!res.ok) {
            console.error('History fetch failed:', res.status, res.statusText);
            // Generate simulated history for public users
            if (isPublic) {
                const simHistory = generateSimulatedHistory();
                renderChart(simHistory);
                renderLog(simHistory, simHistory.length);
            }
            return;
        }
        const data = await res.json();
        console.log('History data received:', data.readings ? data.readings.length : 0, 'records');
        renderChart(data.readings);
        renderLog(data.readings, data.total);
    } catch (e) {
        console.error('fetchHistory error:', e);
        // Generate simulated history on error for public users
        if (isPublic) {
            const simHistory = generateSimulatedHistory();
            renderChart(simHistory);
            renderLog(simHistory, simHistory.length);
        }
    }
}

// Generate simulated history data for demonstration
function generateSimulatedHistory() {
    const readings = [];
    const now = new Date();
    const points = currentRange === '7d' ? 168 : 24; // hourly points
    for (let i = points; i >= 0; i--) {
        const t = new Date(now.getTime() - i * 3600 * 1000);
        readings.push({
            timestamp: t.toISOString(),
            temperature: 25 + Math.sin(i / 5) * 3 + Math.random() * 2,
            moisture: 50 + Math.cos(i / 8) * 10 + Math.random() * 5,
            ec: 1300 + Math.sin(i / 10) * 200 + Math.random() * 100,
            ph: 7.0 + Math.sin(i / 12) * 0.5 + Math.random() * 0.3,
            nitrogen: 90 + Math.random() * 20,
            phosphorus: 60 + Math.random() * 15,
            potassium: 170 + Math.random() * 40
        });
    }
    return readings;
}

function setRange(range) {
    currentRange = range;
    document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.ctrl-btn[data-range="${range}"]`).classList.add('active');
    fetchHistory();
}

/**
 * Aggregate multi-sensor readings by timestamp.
 * When 2+ sensors poll at the same time, average their values
 * so the chart shows a smooth trend instead of zigzag oscillation.
 */
function aggregateReadings(readings) {
    const groups = {};
    for (const r of readings) {
        const key = r.timestamp;
        if (!groups[key]) {
            groups[key] = { ...r, _count: 1 };
        } else {
            const g = groups[key];
            g._count++;
            for (const f of ['temperature', 'moisture', 'ec', 'ph', 'nitrogen', 'phosphorus', 'potassium']) {
                if (r[f] != null && g[f] != null) g[f] += r[f];
            }
        }
    }
    return Object.values(groups).map(g => {
        const n = g._count;
        for (const f of ['temperature', 'moisture', 'ec', 'ph', 'nitrogen', 'phosphorus', 'potassium']) {
            if (g[f] != null) g[f] = +(g[f] / n).toFixed(2);
        }
        delete g._count;
        return g;
    });
}

function renderChart(readings) {
    console.log('renderChart called with', readings ? readings.length : 0, 'readings');
    if (!readings || readings.length === 0) {
        console.warn('No readings data for chart');
        // Clear the chart so stale data from a different range doesn't persist
        if (historyChart) { historyChart.destroy(); historyChart = null; }
        return;
    }

    // Reverse to chronological, then aggregate multi-sensor data
    const chronological = [...readings].reverse();
    const sorted = aggregateReadings(chronological);

    const labels = sorted.map(r => {
        const d = new Date(r.timestamp);
        return currentRange === '7d'
            ? d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    });

    const datasets = [
        { label: 'Temp °C',    data: sorted.map(r => r.temperature), borderColor: '#ef4444', yAxisID: 'y' },
        { label: 'Moisture %', data: sorted.map(r => r.moisture),    borderColor: '#3b82f6', yAxisID: 'y' },
        { label: 'pH',         data: sorted.map(r => r.ph),          borderColor: '#a855f7', yAxisID: 'y' },
        { label: 'EC µS/cm',   data: sorted.map(r => r.ec),          borderColor: '#f59e0b', yAxisID: 'y1', borderDash: [4, 2] },
        { label: 'N mg/kg',    data: sorted.map(r => r.nitrogen),    borderColor: '#22c55e', yAxisID: 'y1' },
        { label: 'P mg/kg',    data: sorted.map(r => r.phosphorus),  borderColor: '#06b6d4', yAxisID: 'y1' },
        { label: 'K mg/kg',    data: sorted.map(r => r.potassium),   borderColor: '#f97316', yAxisID: 'y1' },
    ];
    datasets.forEach(ds => { ds.fill = false; ds.tension = 0.35; ds.pointRadius = 1; ds.borderWidth = 1.5; });

    const ctx = document.getElementById('historyChart').getContext('2d');

    if (historyChart) {
        historyChart.data.labels = labels;
        historyChart.data.datasets = datasets;
        historyChart.update('none');
        return;
    }

    historyChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            animation: { duration: 400 },
            plugins: {
                legend: {
                    position: 'top', align: 'end',
                    labels: { color: '#8899aa', usePointStyle: true, pointStyle: 'circle', padding: 14, font: { size: 11 } },
                },
                tooltip: {
                    backgroundColor: '#131a24', borderColor: '#1e2d3d', borderWidth: 1,
                    titleColor: '#e2e8f0', bodyColor: '#8899aa', titleFont: { size: 12 }, bodyFont: { size: 11 },
                    padding: 10, cornerRadius: 6,
                },
            },
            scales: {
                x: {
                    ticks: { color: '#556677', maxTicksLimit: 10, maxRotation: 0, font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    position: 'left',
                    title: { display: true, text: 'Temp / Moisture / pH', color: '#556677', font: { size: 10 } },
                    ticks: { color: '#556677', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'EC / NPK (mg/kg)', color: '#556677', font: { size: 10 } },
                    ticks: { color: '#556677', font: { size: 10 } },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });
}

// ─── Data Log ───
function renderLog(readings, total) {
    console.log('renderLog called with', readings ? readings.length : 0, 'readings, total:', total);
    const tbody = document.getElementById('log-body');
    const countEl = document.getElementById('log-count');
    if (!tbody) {
        console.warn('log-body element not found');
        return;
    }
    if (!readings) {
        console.warn('No readings data for log');
        return;
    }
    if (countEl) countEl.textContent = `${total || readings.length} records`;

    tbody.innerHTML = readings.slice(0, 50).map(r => {
        const t = new Date(r.timestamp).toLocaleString([], {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
        return `<tr>
            <td class="mono">${r.sensor_id || '--'}</td>
            <td>${t}</td>
            <td>${r.temperature != null ? r.temperature.toFixed(1) : '--'}</td>
            <td>${r.moisture != null ? r.moisture.toFixed(1) : '--'}</td>
            <td>${r.ec != null ? Math.round(r.ec) : '--'}</td>
            <td>${r.ph != null ? r.ph.toFixed(2) : '--'}</td>
            <td>${r.nitrogen != null ? Math.round(r.nitrogen) : '--'}</td>
            <td>${r.phosphorus != null ? Math.round(r.phosphorus) : '--'}</td>
            <td>${r.potassium != null ? Math.round(r.potassium) : '--'}</td>
        </tr>`;
    }).join('');
}

// ─── Manual Read ───
async function triggerRead() {
    const btn = document.getElementById('btn-read');
    const origHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span>Reading...</span>';
    try {
        const res = await fetch(`${API}${API_PREFIX}/read`, { method: 'POST' });
        const data = await res.json();
        if (data.success && data.reading) updateCards(data.reading);
        btn.innerHTML = data.success ? '<span>Done ✓</span>' : '<span>Failed</span>';
    } catch (e) {
        btn.innerHTML = '<span>Error</span>';
    }
    setTimeout(() => { btn.innerHTML = origHTML; btn.disabled = false; }, 2000);
}

// ─── Refresh Data Log ───
async function refreshLog() {
    const btn = document.querySelector('.btn-refresh-log');
    if (btn) { btn.classList.add('spinning'); btn.disabled = true; }
    await fetchHistory();
    if (btn) { setTimeout(() => { btn.classList.remove('spinning'); btn.disabled = false; }, 400); }
}

// ─── Weather ───
const WEATHER_ICONS = {
    0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
    45: '🌫️', 48: '🌫️',
    51: '🌦️', 53: '🌧️', 55: '🌧️',
    61: '🌧️', 63: '🌧️', 65: '🌧️',
    71: '🌨️', 73: '🌨️', 75: '❄️',
    80: '🌦️', 81: '🌧️', 82: '⛈️',
    95: '⛈️', 96: '⛈️', 99: '⛈️',
};

async function fetchWeather() {
    try {
        const res = await fetch(`${API}/api/weather`);
        if (!res.ok) { console.warn('Weather API:', res.status); return; }
        const w = await res.json();

        // Track active city
        if (w.city) currentCity = w.city;

        const icon = WEATHER_ICONS[w.weather_code] || '🌡️';
        document.getElementById('weather-icon').textContent = icon;
        document.getElementById('weather-desc').textContent =
            `${w.city} — ${w.description}`;

        const parts = [];
        if (w.temperature != null) parts.push(`${w.temperature.toFixed(1)}°C`);
        if (w.humidity != null) parts.push(`Humidity ${w.humidity}%`);
        if (w.wind_speed) parts.push(`Wind ${w.wind_speed} km/h`);
        if (w.is_raining) parts.push(`Rain ${w.rain}mm`);
        document.getElementById('weather-detail').textContent = parts.join(' · ');

        // Show soil type badge
        const soilEl = document.getElementById('soil-badge');
        if (soilEl && w.soil_type) {
            soilEl.textContent = `🌱 ${w.soil_type}`;
            soilEl.title = w.soil_description || '';
            soilEl.style.display = '';
        } else if (soilEl) {
            soilEl.style.display = 'none';
        }
    } catch (e) { console.error('fetchWeather error:', e); }
}

async function fetchCities() {
    try {
        const res = await fetch(`${API}/api/weather/cities`);
        if (!res.ok) { console.warn('Cities API:', res.status); return; }
        const data = await res.json();
        citiesData = data.cities;
        // Only use server default if no city is already selected (from localStorage)
        if (!currentCity) {
            currentCity = data.current || '';
            console.log('Using server default city:', currentCity);
        } else {
            console.log('Preserving user selected city:', currentCity);
        }
        const sel = document.getElementById('city-select');
        sel.innerHTML = data.cities.map(c => {
            const status = c.is_active ? '🟢' : '🔴';
            return `<option value="${c.name}" ${c.name === currentCity ? 'selected' : ''}>${status} ${c.name} — ${c.soil_type || c.state}</option>`;
        }).join('');
        // Update toggle button state
        updateToggleBtn();
    } catch (e) { console.error('fetchCities error:', e); }
}

function updateToggleBtn() {
    const btn = document.getElementById('btn-toggle-sensor');
    if (!btn) return;
    const city = citiesData.find(c => c.name === currentCity);
    if (city) {
        btn.textContent = city.is_active ? '⏻ ON' : '⏻ OFF';
        btn.className = 'btn-toggle-sensor ' + (city.is_active ? 'sensor-on' : 'sensor-off');
        btn.title = city.is_active ? `${city.name} sensor is active — click to turn off` : `${city.name} sensor is inactive — click to turn on`;
    }
}

async function toggleSensor() {
    const city = citiesData.find(c => c.name === currentCity);
    if (!city) return;
    const btn = document.getElementById('btn-toggle-sensor');
    if (btn) btn.disabled = true;
    try {
        const res = await fetch(`${API}${API_PREFIX.replace('/sensor','')}/locations/${city.id}/toggle`, { method: 'PATCH' });
        if (res.ok) {
            await fetchCities();
            await Promise.allSettled([fetchLive(), fetchHistory()]);
        }
    } catch (e) { console.error('toggleSensor error:', e); }
    if (btn) btn.disabled = false;
}

async function changeCity(city) {
    if (!city) return;
    const sel = document.getElementById('city-select');
    sel.disabled = true;
    try {
        // Fix endpoint - weather API is at /api/weather, not /api/sensor/weather
        const res = await fetch(`${API}/api/weather/city`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city }),
        });
        if (res.ok) {
            currentCity = city;
            // Save to localStorage so it persists across refreshes
            localStorage.setItem('selectedCity', city);
            updateToggleBtn();
            // Refresh everything for the new city
            await Promise.allSettled([fetchWeather(), fetchLive(), fetchHistory()]);
        } else {
            console.warn('Change city failed:', res.status);
        }
    } catch (e) { console.error('changeCity error:', e); }
    sel.disabled = false;
}

// ─── Add City Modal ───
function openAddCityModal() {
    document.getElementById('add-city-modal').style.display = 'flex';
    document.getElementById('ac-name').focus();
}
function closeAddCityModal() {
    document.getElementById('add-city-modal').style.display = 'none';
    document.getElementById('ac-msg').textContent = '';
}
async function submitAddCity(e) {
    e.preventDefault();
    const btn = document.getElementById('ac-submit');
    const msg = document.getElementById('ac-msg');
    btn.disabled = true;
    btn.textContent = 'Adding...';
    msg.textContent = '';
    const body = {
        name: document.getElementById('ac-name').value.trim(),
        state: document.getElementById('ac-state').value.trim(),
        lat: parseFloat(document.getElementById('ac-lat').value),
        lon: parseFloat(document.getElementById('ac-lon').value),
        soil_type: document.getElementById('ac-soil').value,
        num_sensors: parseInt(document.getElementById('ac-sensors').value) || 2,
    };
    try {
        const res = await fetch(`${API}${API_PREFIX.replace('/sensor','')}/locations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (res.ok) {
            msg.style.color = '#22c55e';
            msg.textContent = `✓ ${body.name} added successfully!`;
            // Refresh city dropdown
            await fetchCities();
            setTimeout(closeAddCityModal, 1200);
        } else {
            const err = await res.json().catch(() => ({}));
            msg.style.color = '#ef4444';
            msg.textContent = err.detail || `Error ${res.status}`;
        }
    } catch (err) {
        msg.style.color = '#ef4444';
        msg.textContent = 'Network error';
    }
    btn.disabled = false;
    btn.textContent = 'Add City';
}

// ─── Startup ───
async function init() {
    console.log('Dashboard initializing...');
    console.log('Sensor API endpoint:', API + API_PREFIX);
    console.log('Health API endpoint:', API + '/api/health');
    
    // Show developer warning on public URL (after DOM is ready)
    if (isPublic) {
        const warningEl = document.getElementById('dev-warning');
        if (warningEl) warningEl.style.display = 'block';
        console.log('Public URL detected - will use simulation mode if API fails');
    }
    
    // Load saved city from localStorage (for persistence across refreshes)
    const savedCity = localStorage.getItem('selectedCity');
    if (savedCity) {
        console.log('Loaded saved city from localStorage:', savedCity);
        currentCity = savedCity;
    }
    
    // 1. Fetch city list + weather FIRST so currentCity is set
    console.log('Step 1: Fetching cities, weather, and health...');
    await Promise.allSettled([fetchCities(), fetchWeather(), fetchHealth()]);

    // 2. Now fetch data with the correct city filter
    console.log('Step 2: Fetching live data and history...');
    await Promise.allSettled([fetchLive(), fetchHistory()]);
    
    // 3. Check if API connection failed and show warning
    const healthStatus = document.getElementById('sensor-status')?.textContent;
    if (healthStatus === 'Offline' || healthStatus === 'Demo Mode') {
        const apiErrorEl = document.getElementById('api-error-warning');
        if (apiErrorEl) apiErrorEl.style.display = 'block';
        
        // For public URLs with no API, immediately show simulation
        if (isPublic) {
            console.log('API unreachable - activating simulation mode');
            isSimulationMode = true;
            // Generate initial simulation data to prevent blank page
            const simData = generateSimulatedData();
            updateCards(simData);
            const simHistory = generateSimulatedHistory();
            renderChart(simHistory);
            renderLog(simHistory, simHistory.length);
        }
    }
    
    console.log('Dashboard initialization complete');

    // Refresh data at poll interval; weather every 5 min
    setInterval(() => { fetchHealth(); fetchLive(); fetchHistory(); }, REFRESH_MS);
    setInterval(fetchWeather, 300000);
}

document.addEventListener('DOMContentLoaded', init);
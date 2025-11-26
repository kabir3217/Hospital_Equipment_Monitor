
import flask
from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import time
import os

app = Flask(__name__)

DEVICE_MAPPING = {
    "ECG Monitor": 0,
    "Ventilator": 1,
    "Infusion Pump": 2,
    "Ultrasound Scanner": 3,
    "X-Ray Machine": 4,
    "Defibrillator": 5,
    "Patient Monitor": 6,
    "Anesthesia Machine": 7
}

MODEL_PATH = "trained_rul_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Put trained_rul_model.pkl here.")

model = joblib.load(MODEL_PATH)

def predict_rul(device_name, usage_hours, temperature, error_count):
    """
    Returns predicted RUL in YEARS and a label ('High'/'Medium'/'Low') and status message.
    Model expects features in order: ['device_code','usage_hours','temperature','error_count']
    and returns RUL in same units used at training (we convert to years by dividing by 365 if it's days).
    """
    device_code = DEVICE_MAPPING.get(device_name, 0)
    X = np.array([[device_code, float(usage_hours), float(temperature), int(error_count)]])
    pred = model.predict(X)[0] 
    rul_years = float(pred) / 365.0
   
    if rul_years > 5:
        label = "High"      
        status = "✅ Healthy — No action needed"
        color = "#198754"   
    elif rul_years > 2:
        label = "Medium"    
        status = "⚠️ Moderate — Schedule maintenance soon"
        color = "#fd7e14"   
    else:
        label = "Low"       
        status = "🔴 Critical — Immediate maintenance required!"
        color = "#dc3545"  
    return {
        "rul_years": round(rul_years, 3),
        "label": label,
        "status": status,
        "color": color,
        "raw_prediction": float(pred)
    }

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Medical Machine Prediction Model</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <!-- Minimal inline CSS for a light medical tone -->
  <style>
    :root{
      --bg:#f6fbfc;
      --card:#ffffff;
      --accent:#0d6efd;
      --soft:#e9f7fb;
      --muted:#6c757d;
      --success:#198754;
      --danger:#dc3545;
      --warm:#fd7e14;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    body{ background:var(--bg); color:#0b2b36; margin:0; padding:24px;}
    .container{ max-width:1100px; margin:0 auto; }
    header{ display:flex; align-items:center; gap:16px; margin-bottom:18px;}
    .brand{ display:flex; flex-direction:column; }
    h1{ margin:0; font-size:20px; color: #063244; }
    p.lead{ margin:0; color:var(--muted); font-size:13px }
    .grid{ display:grid; grid-template-columns: 360px 1fr; gap:18px; align-items:start; }
    .card{ background:var(--card); border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(6,18,22,0.04); }
    label{ font-size:13px; display:block; margin-bottom:6px; color:#0b2b36; }
    select, input[type=range], input[type=number] { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #e1eef2; background:transparent; }
    .small{ font-size:12px; color:var(--muted); margin-top:6px; }
    button{ border:0; padding:10px 14px; border-radius:10px; cursor:pointer; background:var(--accent); color:white; font-weight:600; }
    .muted{ color:var(--muted); font-size:13px }
    .status { font-weight:700; padding:8px 12px; border-radius:10px; display:inline-block; color:white; }
    #chart-wrap{ height:320px; }
    .row{ display:flex; gap:12px; }
    .col{ flex:1; }
    footer{ margin-top:18px; color:var(--muted); font-size:13px; text-align:center; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="container">
    <header>
      <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,var(--soft),#e0f7ff);display:flex;align-items:center;justify-content:center;">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2v20" stroke="#0d6efd" stroke-width="1.6" stroke-linecap="round"/><path d="M5 7h14" stroke="#0d6efd" stroke-width="1.6" stroke-linecap="round"/></svg>
      </div>
      <div class="brand">
        <h1>Medical Machine Prediction Model</h1>
        <p class="lead">Interactive predictive maintenance • light medical UI</p>
      </div>
    </header>

    <div class="grid">
      <!-- Left controls -->
      <div class="card">
        <label for="device">Device</label>
        <select id="device">
          {% for d in devices %}
            <option>{{d}}</option>
          {% endfor %}
        </select>

        <div style="margin-top:12px;">
          <label>Usage Hours</label>
          <input id="usage" type="range" min="0" max="10000" step="1" value="1200">
          <div class="row" style="margin-top:6px;">
            <div class="col"><input id="usage_num" type="number" min="0" max="100000" step="1" value="1200"></div>
            <div class="col muted small">hours</div>
          </div>
        </div>

        <div style="margin-top:12px;">
          <label>Temperature (°C)</label>
          <input id="temp" type="range" min="0" max="120" step="0.1" value="45.0">
          <div class="row" style="margin-top:6px;">
            <div class="col"><input id="temp_num" type="number" min="0" max="150" step="0.1" value="45.0"></div>
            <div class="col muted small">°C</div>
          </div>
        </div>

        <div style="margin-top:12px;">
          <label>Error Count</label>
          <input id="errors" type="range" min="0" max="50" step="1" value="1">
          <div class="row" style="margin-top:6px;">
            <div class="col"><input id="errors_num" type="number" min="0" max="100" step="1" value="1"></div>
            <div class="col muted small">errors</div>
          </div>
        </div>

        <div style="margin-top:14px;" class="row">
          <button id="predictBtn">Predict Now</button>
          <button id="simulateBtn" style="background:#0dcaf0">Start Simulation</button>
        </div>

        <div style="margin-top:12px;">
          <div class="muted small">Model file: <strong>{{model_name}}</strong></div>
          <div class="muted small">Tip: Use <em>Simulate</em> to auto-generate varying inputs.</div>
        </div>

        <div style="margin-top:16px;">
          <div>Prediction:</div>
          <div id="predictionBox" style="margin-top:8px;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div id="statusBadge" class="status" style="background:#198754">—</div>
              <div>
                <div id="rulText" style="font-weight:700; font-size:16px;">—</div>
                <div id="labelText" class="muted small">—</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Chart & history -->
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <h3 style="margin:0;">Live RUL & Sensor Chart</h3>
            <div class="muted small">Shows history of simulated/predicted values</div>
          </div>
          <div class="muted small">Last update: <span id="lastUpdate">—</span></div>
        </div>

        <div id="chart-wrap" style="margin-top:10px;">
          <canvas id="rulChart"></canvas>
        </div>

        <div style="margin-top:12px;">
          <h4 style="margin:0 0 8px 0;">History (latest first)</h4>
          <div id="historyList" class="muted small">No history yet.</div>
        </div>
      </div>
    </div>

    <footer>
      Model inference runs locally on the server. Ensure trained_rul_model.pkl matches the training feature order.
    </footer>
  </div>

<script>
  // DOM elements
  const usage = document.getElementById('usage');
  const usage_num = document.getElementById('usage_num');
  const temp = document.getElementById('temp');
  const temp_num = document.getElementById('temp_num');
  const errors = document.getElementById('errors');
  const errors_num = document.getElementById('errors_num');

  // sync sliders and numbers
  function syncRange(range, num){
    range.addEventListener('input', ()=>{ num.value = range.value; });
    num.addEventListener('input', ()=>{ range.value = num.value; });
  }
  syncRange(usage, usage_num);
  syncRange(temp, temp_num);
  syncRange(errors, errors_num);

  const deviceSelect = document.getElementById('device');
  const predictBtn = document.getElementById('predictBtn');
  const simulateBtn = document.getElementById('simulateBtn');
  const statusBadge = document.getElementById('statusBadge');
  const rulText = document.getElementById('rulText');
  const labelText = document.getElementById('labelText');
  const lastUpdate = document.getElementById('lastUpdate');
  const historyList = document.getElementById('historyList');

  // Chart.js setup
  const ctx = document.getElementById('rulChart').getContext('2d');
  const chartData = {
    labels: [],
    datasets: [
      { label: 'RUL (years)', data: [], fill:false, tension:0.2, borderWidth:2 },
      { label: 'Temperature (°C)', data: [], fill:false, tension:0.2, borderWidth:1, borderDash:[5,5] },
      { label: 'Usage Hours', data: [], fill:false, tension:0.2, borderWidth:1, borderDash:[2,2] },
    ]
  };
  const rulChart = new Chart(ctx, {
    type: 'line',
    data: chartData,
    options: {
      interaction:{ mode:'index', intersect:false },
      plugins:{ legend:{display:true} },
      scales:{
        y:{
          beginAtZero:true,
        }
      }
    }
  });

  let history = [];
  function addHistory(entry){
    history.unshift(entry);
    if(history.length>30) history.pop();
    // update list
    historyList.innerHTML = history.map(h=>`<div><strong>${h.time}</strong> — RUL: ${h.rul}y — ${h.status}</div>`).join('');
  }

  // server predict
  async function getPrediction(payload){
    const resp = await fetch('/predict', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    return await resp.json();
  }

  predictBtn.addEventListener('click', async ()=>{
    const payload = {
      device: deviceSelect.value,
      usage_hours: usage.value,
      temperature: temp.value,
      error_count: errors.value
    };
    const res = await getPrediction(payload);
    handlePredictionResponse(res, payload);
  });

  function handlePredictionResponse(res, payload){
    const now = new Date().toLocaleTimeString();
    lastUpdate.textContent = now;
    statusBadge.textContent = res.status;
    statusBadge.style.background = res.color;
    rulText.textContent = res.rul_years + " years remaining";
    labelText.textContent = `${res.label} — ${res.raw_prediction.toFixed(1)} (raw)`;
    // push to chart
    chartData.labels.push(now);
    chartData.datasets[0].data.push(res.rul_years);
    chartData.datasets[1].data.push(parseFloat(payload.temperature));
    chartData.datasets[2].data.push(parseFloat(payload.usage_hours));
    // keep last 40 points
    if(chartData.labels.length>40){
      chartData.labels.shift();
      chartData.datasets.forEach(ds=>ds.data.shift());
    }
    rulChart.update();
    addHistory({time: now, rul: res.rul_years, status: res.status});
  }

  // Simulation
  let simInterval = null;
  simulateBtn.addEventListener('click', ()=>{
    if(simInterval){
      clearInterval(simInterval); simInterval=null; simulateBtn.textContent='Start Simulation';
    } else {
      simulateBtn.textContent='Stop Simulation';
      simInterval = setInterval(async ()=>{
        // random-ish walk near current values
        usage.value = Math.max(0, Number(usage.value) + (Math.random()*200 - 100));
        usage_num.value = usage.value;
        temp.value = Math.max(0, Number(temp.value) + (Math.random()*4 - 2));
        temp_num.value = temp.value;
        errors.value = Math.max(0, Math.round(Number(errors.value) + (Math.random()*3 - 1)));
        errors_num.value = errors.value;
        // call predict
        const payload = {
          device: deviceSelect.value,
          usage_hours: usage.value,
          temperature: temp.value,
          error_count: errors.value
        };
        const res = await getPrediction(payload);
        handlePredictionResponse(res, payload);
      }, 1800);
    }
  });

  // initial prediction on load
  window.addEventListener('load', ()=>{
    predictBtn.click();
  });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE, devices=list(DEVICE_MAPPING.keys()), model_name=os.path.basename(MODEL_PATH))

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    device = data.get("device", list(DEVICE_MAPPING.keys())[0])
    usage_hours = data.get("usage_hours", 1000)
    temperature = data.get("temperature", 40.0)
    error_count = data.get("error_count", 0)
    try:
        result = predict_rul(device, usage_hours, temperature, error_count)
        result['time'] = time.strftime("%H:%M:%S")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5001, debug=True)
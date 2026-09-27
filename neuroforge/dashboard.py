from fastapi.responses import HTMLResponse

DASHBOARD_HTML="""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroForge Operations</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;background:#f6f7f9;color:#17191c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px}.card{background:white;border:1px solid #ddd;border-radius:14px;padding:18px}.value{font-size:28px;font-weight:700;margin-top:8px}
table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #eee}
.badge{display:inline-block;padding:4px 8px;border-radius:99px;background:#eee}
</style></head><body>
<h1>NeuroForge Operations</h1><p>Live service telemetry from the local API.</p>
<div id="cards" class="grid"></div><h2>Recent Alerts</h2><div id="alerts">Loading...</div>
<script>
async function load(){
 const [m,a]=await Promise.all([fetch('/metrics').then(r=>r.json()),fetch('/observability/alerts?limit=10').then(r=>r.json())]);
 const cards=[['Status',m.health.status],['Predictions',m.api.predictions_total],['Failures',m.api.prediction_failures_total],['Failure rate',(m.api.failure_rate*100).toFixed(2)+'%'],['Mean latency',m.inference.mean_latency_ms.toFixed(2)+' ms'],['Samples in window',m.inference.samples]];
 document.getElementById('cards').innerHTML=cards.map(x=>'<div class="card"><div>'+x[0]+'</div><div class="value">'+x[1]+'</div></div>').join('');
 document.getElementById('alerts').innerHTML=a.length?'<table><tr><th>Type</th><th>Severity</th><th>Value</th><th>Threshold</th><th>Time</th></tr>'+a.slice().reverse().map(x=>'<tr><td>'+x.alert_type+'</td><td><span class="badge">'+x.severity+'</span></td><td>'+x.value+'</td><td>'+x.threshold+'</td><td>'+x.created_at+'</td></tr>').join('')+'</table>':'No recent alerts.';
}
load();setInterval(load,10000);
</script></body></html>"""

def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

let statusChart;
let errorChart;

async function loadMetrics() {
  const metrics = await fetch('/api/metrics').then(r => r.json());
  document.getElementById('total-recipes').textContent = metrics.total_recipes;
  document.getElementById('success-rate').textContent = `${metrics.success_rate_percent}%`;
  document.getElementById('total-cost').textContent = `$${metrics.total_cost_usd}`;
  document.getElementById('avg-latency').textContent = `${metrics.avg_latency_seconds}s`;
  document.getElementById('graph-nodes').textContent = metrics.graph_node_count;
  document.getElementById('total-tokens').textContent = metrics.total_tokens_used.toLocaleString();
  renderStatusChart(metrics);
  renderErrorChart(metrics.error_frequencies);
  const timings = await fetch('/api/metrics/nodes').then(r => r.json());
  document.getElementById('node-timings').innerHTML = timings.map(row => `
    <tr><td>${row.node_name}</td><td>${row.avg_latency_seconds}s</td><td>${row.total_calls}</td><td>${row.total_tokens}</td></tr>
  `).join('');
}

function renderStatusChart(metrics) {
  const data = [metrics.passed_first_try, metrics.passed_after_correction, metrics.failed_unresolvable];
  if (statusChart) statusChart.destroy();
  statusChart = new Chart(document.getElementById('status-chart'), {
    type: 'doughnut',
    data: { labels: ['PASSED', 'CORRECTED', 'FAILED'], datasets: [{ data, backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'] }] },
    options: { plugins: { legend: { labels: { color: '#f1f5f9' } } } }
  });
}

function renderErrorChart(errors) {
  if (errorChart) errorChart.destroy();
  errorChart = new Chart(document.getElementById('error-chart'), {
    type: 'bar',
    data: { labels: errors.map(e => e.error_type), datasets: [{ data: errors.map(e => e.count), backgroundColor: '#3b82f6' }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } } }
  });
}

document.getElementById('extract-form').addEventListener('submit', submitExtraction);
document.querySelectorAll('.tab-btn').forEach(button => {
  button.addEventListener('click', () => switchTab(button.dataset.tab));
});

const dropZone = document.getElementById('dropZone');
const pdfFile = document.getElementById('pdfFile');
if (dropZone && pdfFile) {
  dropZone.addEventListener('dragover', handleDragOver);
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', handleDrop);
  dropZone.addEventListener('click', () => pdfFile.click());
  pdfFile.addEventListener('change', handleFileSelect);
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(button => button.classList.remove('active'));
  document.getElementById(`tab-${tabName}`).classList.add('active');
  document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
}

async function submitExtraction(event) {
  event.preventDefault();

  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const maxPapers = document.getElementById('maxPapers').value || 10;
  const progressContainer = document.getElementById('progressSection');
  progressContainer.style.display = 'block';
  document.getElementById('submitBtn').disabled = true;
  resetProgress(progressContainer);

  try {
    const response = await fetch('/api/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_papers: parseInt(maxPapers, 10) })
    });
    const { job_id } = await response.json();
    connectProgressSocket(job_id, progressContainer, document.getElementById('submitBtn'));
  } catch (error) {
    appendProgressLog({ stage: 'error', message: `Failed to start extraction: ${error.message}` }, progressContainer);
    document.getElementById('submitBtn').disabled = false;
  }
}

function connectProgressSocket(jobId, progressContainer, submitButton) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/${jobId}`);
  let chunksTotal = 0;
  let chunksDone = 0;

  ws.onmessage = event => {
    const progress = JSON.parse(event.data);
    appendProgressLog(progress, progressContainer);

    if (progress.stage === 'fetched') {
      const match = progress.message.match(/(\d+) text chunks/);
      chunksTotal = match ? parseInt(match[1], 10) : 0;
      updateProgressBar(5, progressContainer);
    }

    if (progress.stage === 'stored') {
      chunksDone += 1;
      const pct = 5 + (chunksDone / Math.max(chunksTotal, 1)) * 90;
      updateProgressBar(pct, progressContainer);
    }

    if (progress.stage === 'complete') {
      updateProgressBar(100, progressContainer);
      if (submitButton) submitButton.disabled = false;
      showCompletionSummary(progress, progressContainer);
      setTimeout(loadMetrics, 2000);
    }

    if (progress.stage === 'error') {
      if (submitButton) submitButton.disabled = false;
    }
  };

  ws.onerror = () => {
    appendProgressLog({ stage: 'error', message: 'WebSocket connection failed' }, progressContainer);
    if (submitButton) submitButton.disabled = false;
  };
}

function resetProgress(progressContainer) {
  progressContainer.querySelector('.progress-log').innerHTML = '';
  updateProgressBar(0, progressContainer);
}

function updateProgressBar(percent, progressContainer) {
  progressContainer.querySelector('.progress-bar').style.width = `${Math.min(Math.round(percent), 100)}%`;
}

function appendProgressLog(progress, progressContainer) {
  const log = progressContainer.querySelector('.progress-log');
  const line = document.createElement('div');
  line.className = `progress-line progress-${progress.stage}`;
  const icons = {
    starting: '🚀',
    fetching: '🔍',
    fetched: '✅',
    extracting: '⚗️',
    stored: '💾',
    complete: '🎉',
    error: '❌'
  };

  line.textContent = `${icons[progress.stage] || '•'} ${progress.message}`;
  if (progress.stage === 'stored' && progress.validation_status) {
    const statusIcon = { PASSED: '✅', CORRECTED: '⚠️', FAILED: '❌' };
    line.textContent = `${statusIcon[progress.validation_status] || '•'} ${progress.recipe_id} - ${progress.validation_status}${progress.correction_count > 0 ? ` (${progress.correction_count} corrections)` : ''}`;
  }

  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function showCompletionSummary(progress, progressContainer) {
  appendProgressLog({
    stage: 'complete',
    message: `${progress.total_recipes || 0} recipes, ${progress.passed || 0} passed, ${progress.corrected || 0} corrected, ${progress.failed || 0} failed`
  }, progressContainer);
}

function handleDragOver(event) {
  event.preventDefault();
  dropZone.classList.add('dragging');
}

function handleDrop(event) {
  event.preventDefault();
  dropZone.classList.remove('dragging');
  if (event.dataTransfer.files.length) {
    pdfFile.files = event.dataTransfer.files;
    showFileInfo(event.dataTransfer.files[0]);
  }
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a PDF file');
    return;
  }
  showFileInfo(file);
}

function showFileInfo(file) {
  const info = document.getElementById('uploadFileInfo');
  info.style.display = 'flex';
  info.style.gap = '12px';
  info.querySelector('.filename').textContent = file.name;
  info.querySelector('.filesize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

async function uploadPDF() {
  const file = document.getElementById('pdfFile').files[0];
  if (!file) return;

  const progressContainer = document.getElementById('uploadProgress');
  const uploadBtn = document.getElementById('uploadBtn');
  const formData = new FormData();
  formData.append('file', file);
  const title = document.getElementById('pdfTitle').value.trim();
  if (title) formData.append('title', title);

  uploadBtn.disabled = true;
  progressContainer.style.display = 'block';
  resetProgress(progressContainer);

  const response = await fetch('/api/upload-pdf', { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json();
    appendProgressLog({ stage: 'error', message: error.detail || 'PDF upload failed' }, progressContainer);
    uploadBtn.disabled = false;
    return;
  }

  const data = await response.json();
  appendProgressLog({ stage: 'fetched', message: `PDF parsed: ${data.chunks_found} sections found` }, progressContainer);
  connectProgressSocket(data.job_id, progressContainer, uploadBtn);
}

async function fetchURL() {
  const url = document.getElementById('paperUrl').value.trim();
  if (!url) return;

  const progressContainer = document.getElementById('urlProgress');
  const fetchBtn = document.getElementById('fetchUrlBtn');
  fetchBtn.disabled = true;
  document.getElementById('paywallMessage').style.display = 'none';
  progressContainer.style.display = 'block';
  resetProgress(progressContainer);

  const title = document.getElementById('urlTitle').value.trim();
  const response = await fetch('/api/fetch-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, title: title || null })
  });
  const data = await response.json();

  if (data.is_paywalled) {
    progressContainer.style.display = 'none';
    document.getElementById('paywallMessage').style.display = 'block';
    document.getElementById('paywallDetails').textContent = data.error;
    document.getElementById('alternatives').innerHTML = `<ul>${data.alternatives.map(item => `<li>${item}</li>`).join('')}</ul>`;
    document.getElementById('paywallPdfTitle').value = document.getElementById('urlTitle').value.trim();
    fetchBtn.disabled = false;
    return;
  }

  if (!response.ok || data.status === 'error') {
    appendProgressLog({ stage: 'error', message: data.detail || data.error || 'URL fetch failed' }, progressContainer);
    fetchBtn.disabled = false;
    return;
  }

  appendProgressLog({ stage: 'fetched', message: `URL fetched: ${data.chunks_found} sections found` }, progressContainer);
  connectProgressSocket(data.job_id, progressContainer, fetchBtn);
}

async function uploadPaywallPDF() {
  const file = document.getElementById('paywallPdfFile').files[0];
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a PDF file');
    return;
  }

  const progressContainer = document.getElementById('urlProgress');
  const paywallPdfBtn = document.getElementById('paywallPdfBtn');
  const formData = new FormData();
  formData.append('file', file);

  const title = document.getElementById('paywallPdfTitle').value.trim();
  if (title) formData.append('title', title);

  paywallPdfBtn.disabled = true;
  progressContainer.style.display = 'block';
  resetProgress(progressContainer);
  appendProgressLog({ stage: 'fetching', message: `Parsing uploaded PDF: ${file.name}` }, progressContainer);

  const response = await fetch('/api/upload-pdf', { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json();
    appendProgressLog({ stage: 'error', message: error.detail || 'PDF upload failed' }, progressContainer);
    paywallPdfBtn.disabled = false;
    return;
  }

  const data = await response.json();
  appendProgressLog({ stage: 'fetched', message: `PDF parsed: ${data.chunks_found} sections found` }, progressContainer);
  connectProgressSocket(data.job_id, progressContainer, paywallPdfBtn);
}

async function pollJob(jobId) {
  const job = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
  if (job.status === 'complete' || job.status === 'failed') {
    await loadMetrics();
    return;
  }
  setTimeout(() => pollJob(jobId), 2000);
}

loadMetrics();
setInterval(loadMetrics, 30000);

function toggleSettings() {
  const body = document.getElementById('settingsBody');
  const toggle = document.getElementById('settingsToggle');
  const expanded = body.style.display === 'none';
  body.style.display = expanded ? 'block' : 'none';
  toggle.textContent = expanded ? '▲' : '▼';
  if (expanded) { loadCurrentSettings(); testAllConnections(); }
}

function selectProvider(provider) {
  document.querySelectorAll('.provider-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.provider === provider));
  ['ollama', 'groq', 'gemini'].forEach(name => { document.getElementById(`${name}-settings`).style.display = name === provider ? 'block' : 'none'; });
}

function showStatus(elementId, connected, message) {
  const element = document.getElementById(elementId);
  if (!element) return;
  element.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
  element.textContent = `${connected ? '✅' : '❌'} ${message}`;
}

async function refreshOllamaModels() {
  const response = await fetch('/api/settings/ollama-models');
  const data = await response.json();
  const select = document.getElementById('ollamaModel');
  select.innerHTML = '';
  if (!data.models.length) { select.innerHTML = '<option value="">No models found</option>'; showStatus('ollamaStatus', false, 'No models found. Run: ollama pull qwen2.5:7b'); return; }
  data.models.forEach(model => { const option = document.createElement('option'); option.value = model; option.textContent = model + (model === data.recommended ? ' (recommended)' : ''); select.appendChild(option); });
  showStatus('ollamaStatus', true, `${data.models.length} models available`);
}

async function testAllConnections() {
  try {
    const response = await fetch('/api/settings/status');
    if (!response.ok) throw new Error('status request failed');
    const status = await response.json();
    showStatus('ollamaStatus', status.ollama.connected, status.ollama.message);
    showStatus('groqStatus', status.groq.connected, status.groq.message);
    showStatus('geminiStatus', status.gemini.connected, status.gemini.message);
    showStatus('neo4jStatus', status.neo4j.connected, `${status.neo4j.message}${status.neo4j.details ? ` — ${status.neo4j.details}` : ''}`);
  } catch (error) { console.error('Connection status failed', error); }
}

async function testProvider(provider, statusId) { await saveSettings(false); const response = await fetch('/api/settings/status'); const status = await response.json(); showStatus(statusId, status[provider].connected, status[provider].message); }
function testGroq() { return testProvider('groq', 'groqStatus'); }
function testGemini() { return testProvider('gemini', 'geminiStatus'); }

async function saveSettings(showSaved = true) {
  const provider = document.querySelector('.provider-tab.active').dataset.provider;
  const backend = document.querySelector('input[name="graphBackend"]:checked').value;
  const payload = {
    llm_provider: provider, graph_backend: backend,
    ollama: { model_name: document.getElementById('ollamaModel').value, base_url: document.getElementById('ollamaUrl').value },
    groq: { api_key: document.getElementById('groqApiKey').value || (document.getElementById('groqApiKey').placeholder === 'API key saved' ? '***' : ''), model_name: document.getElementById('groqModel').value },
    gemini: { api_key: document.getElementById('geminiApiKey').value || (document.getElementById('geminiApiKey').placeholder === 'API key saved' ? '***' : ''), model_name: document.getElementById('geminiModel').value },
    neo4j: { uri: document.getElementById('neo4jUri').value, username: 'neo4j', password: document.getElementById('neo4jPassword').value },
    obsidian: { vault_path: document.getElementById('obsidianPath').value }
  };
  const response = await fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  if (showSaved) { const status = document.getElementById('settingsSaveStatus'); status.textContent = response.ok ? '✅ Settings saved' : '❌ Save failed'; status.style.color = response.ok ? 'var(--accent-green)' : 'var(--accent-red)'; setTimeout(() => { status.textContent = ''; }, 3000); }
  return response.ok;
}

async function loadCurrentSettings() {
  const response = await fetch('/api/settings');
  if (!response.ok) return;
  const settings = await response.json();
  selectProvider(settings.llm_provider);
  document.getElementById('ollamaUrl').value = settings.ollama.base_url;
  document.getElementById('ollamaModel').value = settings.ollama.model_name;
  document.getElementById('groqModel').value = settings.groq.model_name;
  document.getElementById('geminiModel').value = settings.gemini.model_name;
  document.querySelector(`input[name="graphBackend"][value="${settings.graph_backend}"]`).checked = true;
  document.getElementById('obsidianPath').value = settings.obsidian.vault_path;
  document.getElementById('neo4jUri').value = settings.neo4j.uri;
  document.getElementById('neo4jPassword').value = settings.neo4j.password;
  await refreshOllamaModels();
  if (settings.groq.api_key === '***') document.getElementById('groqApiKey').placeholder = 'API key saved';
  if (settings.gemini.api_key === '***') document.getElementById('geminiApiKey').placeholder = 'API key saved';
}

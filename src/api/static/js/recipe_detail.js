function roleClass(role) {
  return `role-badge role-${role.toLowerCase()}`;
}

function quantityText(q) {
  return q ? `${q.value} ${q.unit}` : '-';
}

/** Render a safe external link to the original research paper. */
function renderPaperUrl(rawUrl) {
  const container = document.getElementById('paper-url');
  container.replaceChildren();
  if (!rawUrl) {
    container.textContent = 'No source URL available';
    return;
  }
  try {
    const sourceUrl = new URL(rawUrl);
    if (!['http:', 'https:'].includes(sourceUrl.protocol)) throw new Error('Unsupported URL protocol');
    const link = document.createElement('a');
    link.href = sourceUrl.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Open original paper ↗';
    container.appendChild(link);
  } catch (_error) {
    container.textContent = 'Invalid source URL';
  }
}

async function loadRecipe() {
  const recipeId = window.location.pathname.split('/').pop();
  const recipe = await fetch(`/api/recipes/${recipeId}`).then(r => r.json());
  document.getElementById('recipe-title').textContent = recipe.title || recipe.recipe_id;
  document.getElementById('recipe-subtitle').textContent = `${recipe.validation_status} · ${recipe.recipe_id}`;
  document.getElementById('paper-title').textContent = recipe.source_paper_title || 'Unknown paper';
  document.getElementById('paper-doi').textContent = recipe.source_paper_doi || 'No DOI';
  renderPaperUrl(recipe.source_paper_url);
  document.getElementById('entities-table').innerHTML = recipe.entities.map(e => `
    <tr><td>${e.name}</td><td>${e.formula || '-'}</td><td><span class="${roleClass(e.role)}">${e.role}</span></td><td>${quantityText(e.quantity)}</td><td>${quantityText(e.moles)}</td></tr>
  `).join('');
  const c = recipe.conditions || {};
  document.getElementById('conditions-table').innerHTML = `
    <tr><th>Temperature</th><td>${c.temperature_celsius ?? 'unknown'}°C</td></tr>
    <tr><th>Duration</th><td>${c.duration_hours ?? 'unknown'} hours</td></tr>
    <tr><th>Technique</th><td>${c.technique || 'unknown'}</td></tr>
    <tr><th>Atmosphere</th><td>${c.atmosphere || 'unknown'}</td></tr>
    <tr><th>Pressure</th><td>${c.pressure_atm ?? 'unknown'} atm</td></tr>
  `;
  document.getElementById('validation-section').innerHTML = `<span class="badge">${recipe.validation_status}</span>` + (recipe.correction_history || []).map(r => `<p>Attempt ${r.attempt_number}: ${r.error_type} fixed by ${r.agent_that_fixed}</p>`).join('');
  document.getElementById('observability-section').innerHTML = `
    <p>Tokens: ${recipe.total_tokens_used} · Cost: $${recipe.estimated_cost_usd} · Latency: ${recipe.total_latency_seconds}s</p>
    <p>Model: ${recipe.llm_model}</p>
    <pre>${JSON.stringify({ node_latencies: recipe.node_latencies, node_tokens: recipe.node_tokens }, null, 2)}</pre>
  `;
}

loadRecipe();

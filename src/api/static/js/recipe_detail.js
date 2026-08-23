function roleClass(role) {
  return `role-badge role-${String(role || 'unknown').toLowerCase()}`;
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

function appendTextCell(row, text, tagName = 'td') {
  const cell = document.createElement(tagName);
  cell.textContent = text;
  row.appendChild(cell);
  return cell;
}

function appendConditionRow(table, label, value) {
  const row = document.createElement('tr');
  appendTextCell(row, label, 'th');
  appendTextCell(row, value);
  table.appendChild(row);
}

function renderEntities(entities) {
  const table = document.getElementById('entities-table');
  table.replaceChildren();
  (entities || []).forEach(entity => {
    const row = document.createElement('tr');
    appendTextCell(row, entity.name || 'unknown');
    appendTextCell(row, entity.formula || '-');
    const roleCell = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = roleClass(entity.role);
    badge.textContent = entity.role || 'UNKNOWN';
    roleCell.appendChild(badge);
    row.appendChild(roleCell);
    appendTextCell(row, quantityText(entity.quantity));
    appendTextCell(row, quantityText(entity.moles));
    table.appendChild(row);
  });
}

function renderConditions(conditions) {
  const table = document.getElementById('conditions-table');
  const values = conditions || {};
  table.replaceChildren();
  appendConditionRow(table, 'Temperature', values.temperature_celsius == null ? 'unknown' : `${values.temperature_celsius}°C`);
  appendConditionRow(table, 'Duration', values.duration_hours == null ? 'unknown' : `${values.duration_hours} hours`);
  appendConditionRow(table, 'Technique', values.technique || 'unknown');
  appendConditionRow(table, 'Atmosphere', values.atmosphere || 'unknown');
  appendConditionRow(table, 'Pressure', values.pressure_atm == null ? 'unknown' : `${values.pressure_atm} atm`);
}

function renderValidation(recipe) {
  const validationSection = document.getElementById('validation-section');
  validationSection.replaceChildren();

  const status = document.createElement('span');
  status.className = 'badge';
  status.textContent = recipe.validation_status;
  validationSection.appendChild(status);

  const warnings = recipe.validation_warnings || [];
  if (warnings.length) {
    const heading = document.createElement('h4');
    heading.textContent = `Warnings (${warnings.length})`;
    validationSection.appendChild(heading);
    const list = document.createElement('ul');
    warnings.forEach(warning => {
      const item = document.createElement('li');
      const location = warning.field_path ? ` at ${warning.field_path}` : '';
      item.textContent = `${warning.error_type || 'WARNING'}${location}: ${warning.message || ''}`;
      list.appendChild(item);
    });
    validationSection.appendChild(list);
  }

  (recipe.correction_history || []).forEach(record => {
    const audit = document.createElement('p');
    const routedAgent = record.agent_routed_to || 'unknown agent';
    audit.textContent = `Attempt ${record.attempt_number}: ${record.error_type} routed to ${routedAgent}`;
    validationSection.appendChild(audit);
  });
}

function renderObservability(recipe) {
  const section = document.getElementById('observability-section');
  section.replaceChildren();

  const totals = document.createElement('p');
  totals.textContent = `Tokens: ${recipe.total_tokens_used} · Cost: $${recipe.estimated_cost_usd} · Latency: ${recipe.total_latency_seconds}s`;
  section.appendChild(totals);

  const model = document.createElement('p');
  model.textContent = `Model: ${recipe.llm_model || 'unknown'}`;
  section.appendChild(model);

  const provenance = document.createElement('p');
  provenance.textContent = `Source: ${recipe.source_text_completeness || 'unknown'} · ${recipe.source_acquisition_method || 'unknown'}`;
  section.appendChild(provenance);

  const details = document.createElement('pre');
  details.textContent = JSON.stringify({
    node_latencies: recipe.node_latencies,
    node_tokens: recipe.node_tokens,
    node_extraction_methods: recipe.node_extraction_methods,
  }, null, 2);
  section.appendChild(details);
}

async function loadRecipe() {
  const recipeId = window.location.pathname.split('/').pop();
  const recipe = await fetch(`/api/recipes/${recipeId}`).then(r => r.json());
  document.getElementById('recipe-title').textContent = recipe.title || recipe.recipe_id;
  document.getElementById('recipe-subtitle').textContent = `${recipe.validation_status} · ${recipe.recipe_id}`;
  document.getElementById('paper-title').textContent = recipe.source_paper_title || 'Unknown paper';
  document.getElementById('paper-doi').textContent = recipe.source_paper_doi || 'No DOI';
  renderPaperUrl(recipe.source_paper_url);
  renderEntities(recipe.entities);
  renderConditions(recipe.conditions);
  renderValidation(recipe);
  renderObservability(recipe);
}

loadRecipe();

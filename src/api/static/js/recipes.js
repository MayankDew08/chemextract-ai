let recipes = [];

async function loadRecipes() {
  recipes = await fetch('/api/recipes').then(r => r.json());
  renderRecipes();
}

function renderRecipes() {
  const status = document.getElementById('status-filter').value;
  const search = document.getElementById('recipe-search').value.toLowerCase();
  const sortBy = document.getElementById('sort-by').value;
  let visible = recipes.filter(r => (!status || r.validation_status === status) && ((r.title || '').toLowerCase().includes(search)));
  visible.sort((a, b) => {
    if (sortBy === 'latency') return b.latency_seconds - a.latency_seconds;
    if (sortBy === 'cost') return b.cost_usd - a.cost_usd;
    if (sortBy === 'corrections') return b.correction_count - a.correction_count;
    return b.extracted_at.localeCompare(a.extracted_at);
  });
  document.getElementById('recipe-grid').innerHTML = visible.map(recipe => `
    <article class="recipe-card status-${recipe.validation_status.toLowerCase()}" onclick="location.href='/recipes/${recipe.recipe_id}'">
      <h2>${recipe.validation_status === 'PASSED' ? '✅' : recipe.validation_status === 'CORRECTED' ? '⚠️' : '❌'} ${recipe.title || recipe.recipe_id}</h2>
      <p>${recipe.recipe_id}</p>
      <p>Product: ${recipe.product_names.join(', ') || 'unknown'}</p>
      <p>Conditions: ${recipe.temperature_celsius ?? 'unknown'}°C, ${recipe.duration_hours ?? 'unknown'}h</p>
      <p>Tokens: ${recipe.latency_seconds}s latency · Cost: $${recipe.cost_usd}</p>
      <p>Corrections: ${recipe.correction_count}</p>
    </article>
  `).join('');
}

['status-filter', 'sort-by', 'recipe-search'].forEach(id => document.getElementById(id).addEventListener('input', renderRecipes));
loadRecipes();

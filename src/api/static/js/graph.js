let network;
let nodes;
let edges;

const options = {
  layout: { improvedLayout: true },
  nodes: {
    shape: 'dot',
    borderWidth: 2,
    shadow: { enabled: true, color: 'rgba(0, 0, 0, 0.35)', size: 8, x: 0, y: 2 },
    font: {
      color: '#f8fafc',
      size: 15,
      face: 'Inter, system-ui, sans-serif',
      strokeWidth: 4,
      strokeColor: '#21253a',
    },
    labelHighlightBold: false,
  },
  edges: {
    smooth: { enabled: true, type: 'dynamic', roundness: 0.25 },
    color: { color: '#64748b', highlight: '#93c5fd', hover: '#93c5fd', opacity: 0.45 },
    arrows: { to: { enabled: true, scaleFactor: 0.55 } },
    font: {
      color: '#e2e8f0',
      size: 11,
      strokeWidth: 5,
      strokeColor: '#21253a',
      align: 'middle',
    },
    selectionWidth: 2,
    hoverWidth: 1.5,
  },
  physics: {
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -130,
      centralGravity: 0.006,
      springLength: 210,
      springConstant: 0.035,
      damping: 0.62,
      avoidOverlap: 0.75,
    },
    minVelocity: 0.8,
    stabilization: { enabled: true, iterations: 300, fit: true },
  },
  interaction: { hover: true, tooltipDelay: 150, zoomView: true, dragView: true, multiselect: true }
};

function visibleEdgeLabels(edgeIds) {
  const visibleIds = new Set(edgeIds);
  return edges.get().map(edge => ({
    id: edge.id,
    label: visibleIds.has(edge.id) ? edge.hiddenLabel : '',
  }));
}

function connectedEdgeIds(nodeIds) {
  return nodeIds.flatMap(nodeId => network.getConnectedEdges(nodeId));
}

function shortenGraphLabel(node) {
  if (node.group !== 'reaction' || node.label.length <= 28) return node.label;
  return `${node.label.slice(0, 25)}...`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
}

async function loadGraph() {
  const data = await fetch('/api/graph/data').then(r => r.json());
  nodes = new vis.DataSet(data.nodes.map(node => ({
    ...node,
    title: escapeHtml(node.title || node.label),
    label: shortenGraphLabel(node),
    originalLabel: node.label,
    margin: 8,
  })));
  edges = new vis.DataSet(data.edges.map(edge => ({
    ...edge,
    hiddenLabel: edge.label,
    label: '',
    title: escapeHtml(edge.label),
  })));
  network = new vis.Network(document.getElementById('network'), { nodes, edges }, options);
  network.on('click', params => {
    if (params.nodes.length) showNodeDetails(params.nodes[0], nodes.get(params.nodes[0]));
  });
  network.on('hoverEdge', params => edges.update(visibleEdgeLabels([params.edge])));
  network.on('blurEdge', () => edges.update(visibleEdgeLabels(network.getSelectedEdges())));
  network.on('selectNode', params => edges.update(visibleEdgeLabels(connectedEdgeIds(params.nodes))));
  network.on('deselectNode', () => edges.update(visibleEdgeLabels(network.getSelectedEdges())));
  network.on('selectEdge', params => edges.update(visibleEdgeLabels(params.edges)));
  network.on('deselectEdge', () => edges.update(visibleEdgeLabels([])));
  network.once('stabilized', () => {
    document.getElementById('graph-loading').style.display = 'none';
    network.fit({ animation: { duration: 350, easingFunction: 'easeInOutQuad' } });
  });
  const stats = await fetch('/api/graph/stats').then(r => r.json());
  document.getElementById('graph-stats').innerHTML = `<p>Nodes: ${stats.node_count}</p><p>Edges: ${stats.edge_count}</p><p>Recipes: ${stats.recipe_count}</p>`;
}

async function showNodeDetails(nodeId, nodeData) {
  const label = escapeHtml(nodeData.originalLabel || nodeData.label);
  let html = `<h2>${label}</h2><p>${escapeHtml(nodeData.group)}</p>`;
  if (nodeData.group === 'chemical') {
    const co = await fetch(`/api/graph/query/cooccurrence/${encodeURIComponent(nodeData.originalLabel || nodeData.label)}`).then(r => r.json());
    html += `<p>Role: ${escapeHtml(nodeData.role || 'unknown')}</p><p>Appears in: ${nodeData.reaction_count || 0} reactions</p><h3>Co-occurs with</h3>` + co.slice(0, 5).map(x => `<p>${escapeHtml(x.chemical_name)}: ${x.co_occurrence_count}</p>`).join('');
    const solvents = await fetch(`/api/graph/query/solvents/${encodeURIComponent(nodeData.originalLabel || nodeData.label)}`).then(r => r.json());
    if (solvents.length) html += `<h3>Solvents</h3>` + solvents.slice(0, 5).map(x => `<p>${escapeHtml(x.solvent_name)}: ${x.frequency}</p>`).join('');
  }
  if (nodeData.group === 'reaction') {
    const recipeId = nodeData.recipe_id || nodeId.replace('rxn::', '');
    html += `<p><a href="/recipes/${encodeURIComponent(recipeId)}">View full recipe →</a></p>`;
  }
  document.getElementById('node-info').innerHTML = html;
}

document.getElementById('graph-search').addEventListener('input', event => {
  if (!nodes || !network) return;
  const value = event.target.value.toLowerCase();
  const matches = nodes.get().filter(n => (n.originalLabel || n.label).toLowerCase().includes(value)).map(n => n.id);
  network.selectNodes(matches);
  edges.update(visibleEdgeLabels(connectedEdgeIds(matches)));
  if (matches.length) network.fit({ nodes: matches, animation: { duration: 250 } });
});

loadGraph();

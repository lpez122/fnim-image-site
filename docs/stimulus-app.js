let similarityData = null;
let categories = [];
let models = [];
let conditions = [];
let categoryMap = new Map();
let categoryIndexMap = new Map();
let modelMap = new Map();
let conditionMap = new Map();
let controlsBound = false;

const state = {
  modelId: "",
  selectedCategories: new Set(),
  selectedConditions: new Set(),
  selectedLayers: new Set(),
  activeLayer: "",
  compareMode: false,
  focusCategoryId: "",
  categorySearch: "",
  activeExploration: "neighbors"
};

const bundlePanel = document.querySelector("#bundle-panel");
const conditionGrid = document.querySelector("#condition-grid");
const categoryGrid = document.querySelector("#category-grid");
const layerGrid = document.querySelector("#layer-grid");
const summaryPanel = document.querySelector("#summary-panel");
const neighborsGrid = document.querySelector("#neighbors-grid");
const mapGrid = document.querySelector("#map-grid");
const pairGrid = document.querySelector("#pair-grid");
const matrixGrid = document.querySelector("#matrix-grid");
const compareModeToggle = document.querySelector("#compare-mode");
const activeLayerSelect = document.querySelector("#active-layer-select");
const focusCategorySelect = document.querySelector("#focus-category-select");
const modelSelect = document.querySelector("#model-select");
const categorySearchInput = document.querySelector("#category-search");
const caption = document.querySelector("#viz-caption");
const modelCount = document.querySelector("#model-count");
const layerCount = document.querySelector("#layer-count");
const categoryCount = document.querySelector("#category-count");
const sourceImageCount = document.querySelector("#source-image-count");
const selectionCount = document.querySelector("#selection-count");
const advancedMatrix = document.querySelector("#advanced-matrix");
const explorationTabs = document.querySelectorAll("[data-exploration-tab]");
const explorationPanels = {
  neighbors: document.querySelector("#exploration-panel-neighbors"),
  projection: document.querySelector("#exploration-panel-projection"),
  pairs: document.querySelector("#exploration-panel-pairs")
};

if (
  bundlePanel &&
  conditionGrid &&
  categoryGrid &&
  layerGrid &&
  summaryPanel &&
  neighborsGrid &&
  mapGrid &&
  pairGrid &&
  matrixGrid &&
  compareModeToggle &&
  activeLayerSelect &&
  focusCategorySelect &&
  modelSelect &&
  categorySearchInput &&
  caption &&
  advancedMatrix &&
  explorationTabs.length === 3 &&
  explorationPanels.neighbors &&
  explorationPanels.projection &&
  explorationPanels.pairs
) {
  initializeExplorer();
}

async function initializeExplorer() {
  renderLoadingState();

  try {
    const response = await fetch("data/stimulus-set-data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load similarity data (${response.status})`);
    }

    similarityData = await response.json();
    categories = similarityData.categories ?? [];
    conditions = similarityData.conditions ?? [];
    categoryMap = new Map(categories.map((category) => [category.id, category]));
    categoryIndexMap = new Map(categories.map((category, index) => [category.id, index]));
    conditionMap = new Map(conditions.map((condition) => [condition.id, condition]));

    const orderedModelIds =
      similarityData.modelsOrder ?? Object.keys(similarityData.models ?? {});

    models = orderedModelIds
      .map((modelId) => {
        const model = similarityData.models?.[modelId];
        if (!model) {
          return null;
        }

        return {
          id: modelId,
          ...model,
          layerMap: new Map((model.layers ?? []).map((layer) => [layer.id, layer]))
        };
      })
      .filter(Boolean);

    modelMap = new Map(models.map((model) => [model.id, model]));

    state.selectedCategories = new Set(categories.map((category) => category.id));
    state.selectedConditions = new Set(conditions.map((condition) => condition.id));
    state.focusCategoryId =
      similarityData.defaults?.focusCategoryId && categoryMap.has(similarityData.defaults.focusCategoryId)
        ? similarityData.defaults.focusCategoryId
        : categories[0]?.id ?? "";
    state.compareMode = false;
    compareModeToggle.checked = false;

    const defaultModelId =
      similarityData.defaults?.modelId && modelMap.has(similarityData.defaults.modelId)
        ? similarityData.defaults.modelId
        : models[0]?.id ?? "";

    resetStateForModel(defaultModelId);
    ensureValidFocusCategory();
    populateModelSelect();
    populateFocusCategorySelect();
    populateActiveLayerSelect();
    updateOverviewStats();
    bindControls();
    render();
  } catch (error) {
    renderErrorState(error);
  }
}

function getCurrentModel() {
  return modelMap.get(state.modelId) ?? null;
}

function getLayersForModel(modelId = state.modelId) {
  return modelMap.get(modelId)?.layers ?? [];
}

function getLayerMapForModel(modelId = state.modelId) {
  return modelMap.get(modelId)?.layerMap ?? new Map();
}

function getSelectedCategoryIds() {
  return categories.filter((category) => state.selectedCategories.has(category.id)).map((category) => category.id);
}

function getSelectedLayerIds() {
  return getLayersForModel()
    .filter((layer) => state.selectedLayers.has(layer.id))
    .map((layer) => layer.id);
}

function getCategorySearchResults() {
  const search = state.categorySearch.trim().toLowerCase();
  return categories.filter((category) => {
    if (conditions.length && state.selectedConditions.size === 0) {
      return false;
    }

    if (!state.selectedConditions.has(category.conditionId)) {
      return false;
    }

    if (!search) {
      return true;
    }

    return (
      category.label.toLowerCase().includes(search) ||
      category.id.toLowerCase().includes(search)
    );
  });
}

function resetStateForModel(modelId) {
  state.modelId = modelId;
  const model = getCurrentModel();
  if (!model) {
    state.selectedLayers = new Set();
    state.activeLayer = "";
    return;
  }

  const defaultSelectedLayers = (model.defaults?.selectedLayers ?? []).filter((layerId) =>
    model.layerMap.has(layerId)
  );

  state.selectedLayers = new Set(
    defaultSelectedLayers.length ? defaultSelectedLayers : model.layers.map((layer) => layer.id)
  );

  state.activeLayer =
    model.defaults?.activeLayer && model.layerMap.has(model.defaults.activeLayer)
      ? model.defaults.activeLayer
      : model.layers[0]?.id ?? "";
}

function ensureValidFocusCategory() {
  const selectedCategoryIds = getSelectedCategoryIds();

  if (!selectedCategoryIds.length) {
    state.focusCategoryId = "";
    return;
  }

  if (!state.selectedCategories.has(state.focusCategoryId)) {
    state.focusCategoryId = selectedCategoryIds[0];
  }
}

function bindControls() {
  if (controlsBound) {
    return;
  }

  controlsBound = true;

  conditionGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-condition]");
    if (!button) {
      return;
    }

    const conditionId = button.dataset.condition;
    if (state.selectedConditions.has(conditionId)) {
      state.selectedConditions.delete(conditionId);
    } else {
      state.selectedConditions.add(conditionId);
    }

    renderConditionControls();
    renderCategoryControls();
  });

  categoryGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) {
      return;
    }

    const categoryId = button.dataset.category;
    if (state.selectedCategories.has(categoryId)) {
      state.selectedCategories.delete(categoryId);
    } else {
      state.selectedCategories.add(categoryId);
    }

    ensureValidFocusCategory();
    populateFocusCategorySelect();
    render();
  });

  layerGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-layer]");
    if (!button) {
      return;
    }

    const layerId = button.dataset.layer;
    if (state.selectedLayers.has(layerId) && state.selectedLayers.size > 1) {
      state.selectedLayers.delete(layerId);
      if (state.activeLayer === layerId) {
        state.activeLayer = [...state.selectedLayers][0];
      }
    } else {
      state.selectedLayers.add(layerId);
      state.activeLayer = layerId;
    }

    populateActiveLayerSelect();
    render();
  });

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      const visibleCategoryIds = getCategorySearchResults().map((category) => category.id);

      if (action === "select-all-categories") {
        state.selectedCategories = new Set(visibleCategoryIds.length ? visibleCategoryIds : categories.map((category) => category.id));
      }

      if (action === "clear-categories") {
        state.selectedCategories = new Set();
      }

      if (action === "select-all-conditions" || action === "reset-conditions") {
        state.selectedConditions = new Set(conditions.map((condition) => condition.id));
      }

      if (action === "select-all-layers") {
        state.selectedLayers = new Set(getLayersForModel().map((layer) => layer.id));
      }

      if (action === "reset-layers") {
        resetStateForModel(state.modelId);
      }

      ensureValidFocusCategory();
      populateFocusCategorySelect();
      populateActiveLayerSelect();
      render();
    });
  });

  compareModeToggle.addEventListener("change", (event) => {
    state.compareMode = event.target.checked;
    render();
  });

  activeLayerSelect.addEventListener("change", (event) => {
    state.activeLayer = event.target.value;
    state.selectedLayers.add(state.activeLayer);
    render();
  });

  focusCategorySelect.addEventListener("change", (event) => {
    state.focusCategoryId = event.target.value;
    render();
  });

  modelSelect.addEventListener("change", (event) => {
    resetStateForModel(event.target.value);
    populateActiveLayerSelect();
    updateOverviewStats();
    render();
  });

  categorySearchInput.addEventListener("input", (event) => {
    state.categorySearch = event.target.value;
    renderCategoryControls();
  });

  explorationTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeExploration = tab.dataset.explorationTab;
      renderExplorationTabs();
    });
  });

  advancedMatrix.addEventListener("toggle", () => {
    renderMatrices();
  });
}

function populateModelSelect() {
  modelSelect.innerHTML = models
    .map(
      (model) =>
        `<option value="${model.id}" ${model.id === state.modelId ? "selected" : ""}>${escapeHtml(
          model.label
        )}</option>`
    )
    .join("");
}

function populateFocusCategorySelect() {
  const selectedCategoryIds = getSelectedCategoryIds();

  if (!selectedCategoryIds.length) {
    focusCategorySelect.innerHTML = `<option value="">No groups selected</option>`;
    return;
  }

  focusCategorySelect.innerHTML = selectedCategoryIds
    .map((categoryId) => {
      const category = categoryMap.get(categoryId);
      return `<option value="${categoryId}" ${
        categoryId === state.focusCategoryId ? "selected" : ""
      }>${escapeHtml(category.label)}</option>`;
    })
    .join("");
}

function populateActiveLayerSelect() {
  activeLayerSelect.innerHTML = getLayersForModel()
    .map(
      (layer) =>
        `<option value="${layer.id}" ${layer.id === state.activeLayer ? "selected" : ""}>${escapeHtml(
          layer.label
        )}</option>`
    )
    .join("");
}

function updateOverviewStats() {
  if (modelCount) {
    modelCount.textContent = String(models.length);
  }
  if (layerCount) {
    layerCount.textContent = String(similarityData?.dataset?.conditionCount ?? conditions.length);
  }
  if (categoryCount) {
    categoryCount.textContent = String(similarityData?.dataset?.groupCount ?? categories.length);
  }
  if (sourceImageCount) {
    sourceImageCount.textContent = String(similarityData?.dataset?.totalImages ?? 0);
  }
}

function renderLoadingState() {
  bundlePanel.innerHTML = `
    <div class="bundle-layout">
      <article class="summary-card">
        <h3>Loading stimulus-set overview</h3>
        <p>Preparing condition summaries and download metadata.</p>
      </article>
    </div>
  `;
  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Loading stimulus-set data</h3>
        <p>Reading the precomputed visual and semantic similarity matrices.</p>
      </article>
    </div>
  `;
  neighborsGrid.innerHTML = `<div class="empty-state">Loading nearest-neighbor view.</div>`;
  mapGrid.innerHTML = `<div class="empty-state">Loading 2D similarity projections.</div>`;
  pairGrid.innerHTML = `<div class="empty-state">Loading pair rankings.</div>`;
  matrixGrid.innerHTML = `<div class="empty-state">Loading clustered matrix view.</div>`;
  caption.textContent = "Loading precomputed visual-semantic data.";
}

function renderErrorState(error) {
  const message = error instanceof Error ? error.message : String(error);
  bundlePanel.innerHTML = `
    <div class="bundle-layout">
      <article class="summary-card">
        <h3>Overview unavailable</h3>
        <p>${escapeHtml(message)}</p>
      </article>
    </div>
  `;
  conditionGrid.innerHTML = "";
  categoryGrid.innerHTML = "";
  layerGrid.innerHTML = "";
  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Data load failed</h3>
        <p>${escapeHtml(message)}</p>
        <p>Serve the website over HTTP so the browser can load the generated data file.</p>
      </article>
    </div>
  `;
  neighborsGrid.innerHTML = `
    <div class="empty-state">
      The stimulus set tab could not load <code>data/stimulus-set-data.json</code>.
    </div>
  `;
  mapGrid.innerHTML = neighborsGrid.innerHTML;
  pairGrid.innerHTML = neighborsGrid.innerHTML;
  matrixGrid.innerHTML = neighborsGrid.innerHTML;
  caption.textContent = "The stimulus-set views become available after the data file loads.";
}

function renderBundleOverview() {
  const conditionCards = conditions
    .map(
      (condition) => `
        <article class="bundle-card" style="--chip-accent: ${condition.accent};">
          <h3>${escapeHtml(condition.label)}</h3>
          <p>${escapeHtml(condition.description)}</p>
          <p><strong>${condition.groupCount}</strong> stimulus groups</p>
          <p><strong>${condition.imageCount}</strong> source images</p>
        </article>
      `
    )
    .join("");

  const downloadCards = (similarityData.downloads ?? [])
    .map(
      (download) => `
        <article class="bundle-card">
          <h3>${escapeHtml(download.label)}</h3>
          <p><strong>${escapeHtml(download.sizeLabel)}</strong></p>
          <p>${escapeHtml(download.note)}</p>
          ${
            download.downloadUrl
              ? `<p><a href="${escapeHtml(download.downloadUrl)}">Open download</a></p>`
              : `<p class="selection-note">No public download link is attached yet.</p>`
          }
        </article>
      `
    )
    .join("");

  bundlePanel.innerHTML = `
    <div class="bundle-layout">
      <section>
        <p class="eyebrow">Condition overview</p>
        <div class="bundle-grid">${conditionCards}</div>
      </section>
      <section>
        <p class="eyebrow">Bundle downloads</p>
        <div class="bundle-grid">${downloadCards}</div>
      </section>
    </div>
  `;
}

function renderConditionControls() {
  conditionGrid.innerHTML = conditions
    .map((condition) => {
      const selected = state.selectedConditions.has(condition.id);
      return `
        <button
          type="button"
          class="filter-chip ${selected ? "active" : ""}"
          data-condition="${condition.id}"
          aria-pressed="${selected}"
        >
          ${escapeHtml(condition.label)}
        </button>
      `;
    })
    .join("");
}

function renderCategoryControls() {
  const visibleCategories = getCategorySearchResults();
  const selectedCount = state.selectedCategories.size;

  if (selectionCount) {
    selectionCount.textContent = `${selectedCount} selected / ${visibleCategories.length} visible / ${categories.length} total`;
  }

  if (!visibleCategories.length) {
    categoryGrid.innerHTML = `
      <div class="empty-state compact-empty">
        No stimulus groups match the current filters.
      </div>
    `;
    return;
  }

  categoryGrid.innerHTML = visibleCategories
    .map((category) => {
      const selected = state.selectedCategories.has(category.id);
      const focused = category.id === state.focusCategoryId;
      return `
        <button
          type="button"
          class="category-option ${selected ? "selected" : ""} ${focused ? "focused" : ""}"
          data-category="${category.id}"
          aria-pressed="${selected}"
          style="--chip-accent: ${category.accent};"
        >
          <span class="category-option-swatch"></span>
          <span class="category-option-copy">
            <span class="category-option-title">${escapeHtml(category.label)}</span>
            <span class="category-option-meta">${escapeHtml(category.conditionLabel)} / ${category.sourceCount} images</span>
          </span>
        </button>
      `;
    })
    .join("");
}

function renderLayerControls() {
  layerGrid.innerHTML = getLayersForModel()
    .map((layer) => {
      const selected = state.selectedLayers.has(layer.id);
      return `
        <button
          type="button"
          class="layer-chip ${selected ? "selected" : ""}"
          data-layer="${layer.id}"
          aria-pressed="${selected}"
        >
          <span class="layer-content">
            <span class="layer-topline">
              <span class="layer-title">${escapeHtml(layer.label)}</span>
              <span class="layer-badge">${escapeHtml(layer.stage)}</span>
            </span>
            <span class="layer-meta">${escapeHtml(layer.note)}</span>
          </span>
        </button>
      `;
    })
    .join("");
}

function renderExplorationTabs() {
  explorationTabs.forEach((tab) => {
    const isActive = tab.dataset.explorationTab === state.activeExploration;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  Object.entries(explorationPanels).forEach(([panelId, panel]) => {
    const isActive = panelId === state.activeExploration;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });
}

function render() {
  ensureValidFocusCategory();
  populateFocusCategorySelect();
  renderBundleOverview();
  renderConditionControls();
  renderCategoryControls();
  renderLayerControls();
  renderExplorationTabs();
  updateOverviewStats();
  renderSummary();
  renderNeighbors();
  renderMaps();
  renderPairs();
  renderMatrices();
  updateExplorationCaption();
}

function renderSummary() {
  const categoryIds = getSelectedCategoryIds();
  const model = getCurrentModel();
  const activeLayer = getLayerMapForModel().get(state.activeLayer);

  if (categoryIds.length < 2 || !model || !activeLayer || !state.focusCategoryId) {
    summaryPanel.innerHTML = `
      <div class="summary-grid">
        <article class="summary-card">
          <h3>Selection needed</h3>
          <p>Select at least two stimulus groups to inspect the visual and semantic similarities.</p>
        </article>
      </div>
    `;
    return;
  }

  const totalFolderImages = categoryIds.reduce(
    (sum, categoryId) => sum + (categoryMap.get(categoryId)?.sourceCount ?? 0),
    0
  );
  const totalPublicImages = categoryIds.reduce(
    (sum, categoryId) => sum + (categoryMap.get(categoryId)?.publicCount ?? 0),
    0
  );
  const selectedConditionCount = new Set(
    categoryIds.map((categoryId) => categoryMap.get(categoryId)?.conditionId).filter(Boolean)
  ).size;
  const layerSummary = model.summaries?.[state.activeLayer] ?? null;
  const compareText = state.compareMode
    ? `Comparing ${state.selectedLayers.size} layers or embeddings in ${model.label}.`
    : `Focused on ${activeLayer.label} in ${model.label}.`;

  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Current scope</h3>
        <p><strong>${categoryIds.length}</strong> stimulus groups selected.</p>
        <p><strong>${totalFolderImages}</strong> stimulus images represented.</p>
        <p><strong>${selectedConditionCount}</strong> condition sets represented.</p>
        <p><strong>${escapeHtml(categoryMap.get(state.focusCategoryId)?.label ?? "")}</strong> is the focus group.</p>
      </article>
      <article class="summary-card">
        <h3>Active representation</h3>
        <p><strong>${escapeHtml(model.label)}</strong> ${escapeHtml(activeLayer.label)}</p>
        <p>${escapeHtml(activeLayer.descriptor)}</p>
        <p>${compareText}</p>
      </article>
      <article class="summary-card">
        <h3>Stimulus-set stats</h3>
        <p><strong>${formatScore(layerSummary?.meanOffDiagonal)}</strong> mean off-diagonal group-pair similarity.</p>
        <p><strong>${formatScore(layerSummary?.stdOffDiagonal)}</strong> standard deviation.</p>
        <p>Strongest pair: <strong>${formatPair(layerSummary?.maxPair)}</strong> at ${formatScore(
          layerSummary?.maxValue
        )}.</p>
        <p><strong>${totalPublicImages}</strong> preview thumbnails are available in the current selection.</p>
      </article>
    </div>
  `;
}

function renderNeighbors() {
  const categoryIds = getSelectedCategoryIds();
  const model = getCurrentModel();

  if (categoryIds.length < 2 || !model || !state.focusCategoryId) {
    neighborsGrid.innerHTML = `
      <div class="empty-state">
        Pick at least two stimulus groups to populate the nearest-neighbor view.
      </div>
    `;
    return;
  }

  const layerIdsToRender = state.compareMode ? getSelectedLayerIds() : [state.activeLayer];
  neighborsGrid.innerHTML = layerIdsToRender
    .map((layerId) => renderNeighborCard(model.id, layerId, categoryIds))
    .join("");
}

function renderNeighborCard(modelId, layerId, categoryIds) {
  const model = modelMap.get(modelId);
  const layer = getLayerMapForModel(modelId).get(layerId);
  const focusCategory = categoryMap.get(state.focusCategoryId);
  const neighbors = getTopNeighbors(modelId, layerId, categoryIds, state.focusCategoryId, 6);

  return `
    <article class="neighbor-card">
      <div class="neighbor-header">
        <div>
          <h3>${escapeHtml(layer.label)}</h3>
          <p class="matrix-subtitle">${escapeHtml(model.label)}: ${escapeHtml(layer.note)}</p>
        </div>
        <div class="matrix-metric">${neighbors.length} nearest groups</div>
      </div>

      <div class="focus-preview">
        <div>
          <p class="eyebrow">Focus group</p>
          <h3>${escapeHtml(focusCategory.label)}</h3>
          <p class="sample-category-meta">${escapeHtml(focusCategory.conditionLabel)} / ${focusCategory.sourceCount} images</p>
        </div>
        <div class="focus-thumb-row">
          ${focusCategory.samples
            .map(
              (samplePath, index) => `
                <img
                  class="focus-thumb"
                  src="${samplePath}"
                  alt="${escapeHtml(focusCategory.label)} sample ${index + 1}"
                  loading="lazy"
                />
              `
            )
            .join("")}
        </div>
        <p class="selection-note">${escapeHtml(focusCategory.note)}</p>
      </div>

      <div class="neighbor-list">
        ${neighbors
          .map(
            (neighbor, index) => `
              <div class="neighbor-row">
                <span class="neighbor-rank">${index + 1}</span>
                <img
                  class="neighbor-thumb"
                  src="${neighbor.category.thumbnail}"
                  alt="${escapeHtml(neighbor.category.label)} thumbnail"
                  loading="lazy"
                />
                <div class="neighbor-copy">
                  <strong>${escapeHtml(neighbor.category.label)}</strong>
                  <span>${escapeHtml(neighbor.category.conditionLabel)} / ${neighbor.category.sourceCount} images</span>
                </div>
                <div class="neighbor-bar">
                  <span class="neighbor-bar-fill" style="width: ${Math.max(0, neighbor.score) * 100}%"></span>
                </div>
                <span class="neighbor-score">${neighbor.score.toFixed(2)}</span>
              </div>
            `
          )
          .join("")}
      </div>
    </article>
  `;
}

function renderMaps() {
  const categoryIds = getSelectedCategoryIds();
  const model = getCurrentModel();

  if (categoryIds.length < 2 || !model || !state.focusCategoryId) {
    mapGrid.innerHTML = `
      <div class="empty-state">
        Pick at least two stimulus groups to populate the 2D similarity projection.
      </div>
    `;
    return;
  }

  const layerIdsToRender = state.compareMode ? getSelectedLayerIds() : [state.activeLayer];
  mapGrid.innerHTML = layerIdsToRender
    .map((layerId) => renderMapCard(model.id, layerId, categoryIds))
    .join("");
}

function renderMapCard(modelId, layerId, categoryIds) {
  const model = modelMap.get(modelId);
  const layer = getLayerMapForModel(modelId).get(layerId);
  const neighbors = getTopNeighbors(modelId, layerId, categoryIds, state.focusCategoryId, 5);
  const projectionStats = getProjectionStats(modelId, layerId, categoryIds, state.focusCategoryId);
  const svg = buildProjectionSvg(
    modelId,
    layerId,
    categoryIds,
    state.focusCategoryId,
    neighbors,
    projectionStats.coordinateRange
  );

  return `
    <article class="semantic-card">
      <div class="matrix-header">
        <div>
          <h3>${escapeHtml(layer.label)}</h3>
          <p class="matrix-subtitle">${escapeHtml(model.label)} 2D similarity projection</p>
        </div>
        <div class="matrix-metric">focus ${escapeHtml(categoryMap.get(state.focusCategoryId)?.label ?? "")}</div>
      </div>
      <div class="projection-metrics">
        <p><strong>Selected cosine range</strong> ${formatScore(projectionStats.selectedRange.min)} to ${formatScore(
          projectionStats.selectedRange.max
        )}</p>
        <p><strong>Focus cosine range</strong> ${formatScore(projectionStats.focusRange.min)} to ${formatScore(
          projectionStats.focusRange.max
        )}</p>
        <p><strong>Projection x range</strong> ${formatSigned(projectionStats.coordinateRange.minX)} to ${formatSigned(
          projectionStats.coordinateRange.maxX
        )}</p>
        <p><strong>Projection y range</strong> ${formatSigned(projectionStats.coordinateRange.minY)} to ${formatSigned(
          projectionStats.coordinateRange.maxY
        )}</p>
      </div>
      <p class="projection-note">Axes are arbitrary. Use the raw cosine ranges above to interpret how compressed or spread out the projection really is.</p>
      <div class="map-frame">${svg}</div>
    </article>
  `;
}

function renderPairs() {
  const categoryIds = getSelectedCategoryIds();
  const model = getCurrentModel();

  if (categoryIds.length < 2 || !model) {
    pairGrid.innerHTML = `
      <div class="empty-state">
        Pick at least two stimulus groups to populate the pair ranking.
      </div>
    `;
    return;
  }

  const layerIdsToRender = state.compareMode ? getSelectedLayerIds() : [state.activeLayer];
  pairGrid.innerHTML = layerIdsToRender
    .map((layerId) => renderPairCard(model.id, layerId, categoryIds))
    .join("");
}

function renderPairCard(modelId, layerId, categoryIds) {
  const model = modelMap.get(modelId);
  const layer = getLayerMapForModel(modelId).get(layerId);
  const sections = getPairSections(modelId, layerId, categoryIds);

  return `
    <article class="pair-card">
      <div class="matrix-header">
        <div>
          <h3>${escapeHtml(layer.label)}</h3>
          <p class="matrix-subtitle">${escapeHtml(model.label)} pair ranking across the selected stimulus groups</p>
        </div>
        <div class="matrix-metric">${sections.totalPairs} total pairs</div>
      </div>
      <div class="pair-section-grid">
        ${sections.items.map((section) => renderPairSection(section)).join("")}
      </div>
    </article>
  `;
}

function renderPairSection(section) {
  return `
    <section class="pair-section">
      <div class="pair-section-header">
        <h4>${escapeHtml(section.title)}</h4>
        <span>${section.pairs.length} pairs</span>
      </div>
      <div class="pair-scroll">
        ${section.pairs.map((pair, index) => renderPairRow(pair, index)).join("")}
      </div>
    </section>
  `;
}

function renderPairRow(pair, index) {
  const position = 4 + Math.max(0, Math.min(1, pair.score)) * 92;
  const pairLabel = `${pair.left.label} and ${pair.right.label}`;

  return `
    <div class="pair-row">
      <div class="pair-rank">${index + 1}</div>
      <div class="pair-copy">
        <strong>${escapeHtml(pair.left.label)}</strong>
        <span>${escapeHtml(pair.right.label)}</span>
      </div>
      <div class="pair-track-wrap" aria-label="${escapeHtml(pairLabel)} at ${pair.score.toFixed(2)} cosine">
        <div class="pair-track">
          <span class="pair-axis-label pair-axis-min">0</span>
          <span class="pair-axis-label pair-axis-max">1</span>
          <span class="pair-track-fill" style="width: ${position}%"></span>
          <span class="pair-marker" style="left: ${position}%">
            <img src="${pair.left.thumbnail}" alt="" loading="lazy" />
            <img src="${pair.right.thumbnail}" alt="" loading="lazy" />
          </span>
        </div>
      </div>
      <div class="pair-score">${pair.score.toFixed(2)}</div>
    </div>
  `;
}

function updateExplorationCaption() {
  const categoryIds = getSelectedCategoryIds();
  const layerIdsToRender = state.compareMode ? getSelectedLayerIds() : [state.activeLayer];

  if (categoryIds.length < 2) {
    caption.textContent = "The stimulus-set tab updates from the model, layer, condition, and group controls.";
    return;
  }

  if (state.activeExploration === "neighbors") {
    caption.textContent = state.compareMode
      ? `Showing nearest-neighbor rankings for ${layerIdsToRender.length} layers or embeddings across ${categoryIds.length} selected stimulus groups.`
      : `Showing the nearest-neighbor view for ${categoryIds.length} selected stimulus groups.`;
    return;
  }

  if (state.activeExploration === "projection") {
    caption.textContent = state.compareMode
      ? `Showing 2D similarity projections for ${layerIdsToRender.length} layers or embeddings across ${categoryIds.length} selected stimulus groups.`
      : `Showing the 2D similarity projection for ${categoryIds.length} selected stimulus groups.`;
    return;
  }

  caption.textContent = state.compareMode
    ? `Showing top, middle, and lowest pair rankings for ${layerIdsToRender.length} layers or embeddings across ${categoryIds.length} selected stimulus groups.`
    : `Showing top, middle, and lowest pair rankings for ${categoryIds.length} selected stimulus groups.`;
}

function renderMatrices() {
  const categoryIds = getSelectedCategoryIds();
  const model = getCurrentModel();

  if (!advancedMatrix.open) {
    matrixGrid.innerHTML = `
      <div class="empty-state">
        Open this panel to view the clustered similarity matrix.
      </div>
    `;
    return;
  }

  if (categoryIds.length < 2 || !model) {
    matrixGrid.innerHTML = `
      <div class="empty-state">
        Pick at least two stimulus groups to render the clustered similarity matrix.
      </div>
    `;
    return;
  }

  const layerIdsToRender = state.compareMode ? getSelectedLayerIds() : [state.activeLayer];
  matrixGrid.innerHTML = layerIdsToRender
    .map((layerId) => renderMatrixCard(state.modelId, layerId, categoryIds))
    .join("");
}

function renderMatrixCard(modelId, layerId, categoryIds) {
  const model = modelMap.get(modelId);
  const layer = getLayerMapForModel(modelId).get(layerId);
  const orderedCategoryIds = orderSelectedCategoryIds(modelId, layerId, categoryIds);
  const matrix = buildMatrix(modelId, layerId, orderedCategoryIds);
  const stats = summarizeSubsetMatrix(matrix, orderedCategoryIds);
  const svg = buildMatrixSvg(`${modelId}-${layerId}`, orderedCategoryIds, matrix);

  return `
    <article class="matrix-card">
      <div class="matrix-header">
        <div>
          <h3>${escapeHtml(layer.label)}</h3>
          <p class="matrix-subtitle">${escapeHtml(model.label)} clustered similarity matrix</p>
        </div>
        <div class="matrix-metric">mean cosine ${stats.average.toFixed(2)}</div>
      </div>
      <div class="matrix-frame">${svg}</div>
      <div class="legend">
        <span>lower</span>
        <span class="legend-bar"></span>
        <span>higher</span>
      </div>
    </article>
  `;
}

function getTopNeighbors(modelId, layerId, categoryIds, focusCategoryId, limit) {
  const fullMatrix = modelMap.get(modelId)?.matrices?.[layerId] ?? [];
  const focusIndex = categoryIndexMap.get(focusCategoryId);

  return categoryIds
    .filter((categoryId) => categoryId !== focusCategoryId)
    .map((categoryId) => ({
      id: categoryId,
      score: fullMatrix[focusIndex][categoryIndexMap.get(categoryId)],
      category: categoryMap.get(categoryId)
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}

function orderSelectedCategoryIds(modelId, layerId, categoryIds) {
  const model = modelMap.get(modelId);
  const order = model?.matrixOrders?.[layerId] ?? [];
  const selected = new Set(categoryIds);
  const ordered = order
    .map((index) => categories[index]?.id)
    .filter((categoryId) => categoryId && selected.has(categoryId));
  const missing = categoryIds.filter((categoryId) => !ordered.includes(categoryId));
  return [...ordered, ...missing];
}

function buildMatrix(modelId, layerId, categoryIds) {
  const fullMatrix = modelMap.get(modelId)?.matrices?.[layerId];
  const indices = categoryIds.map((categoryId) => categoryIndexMap.get(categoryId));
  return indices.map((rowIndex) => indices.map((columnIndex) => fullMatrix[rowIndex][columnIndex]));
}

function getSortedPairs(modelId, layerId, categoryIds) {
  const matrix = buildMatrix(modelId, layerId, categoryIds);
  const pairs = [];

  for (let row = 0; row < categoryIds.length; row += 1) {
    for (let column = row + 1; column < categoryIds.length; column += 1) {
      pairs.push({
        score: matrix[row][column],
        left: categoryMap.get(categoryIds[row]),
        right: categoryMap.get(categoryIds[column])
      });
    }
  }

  return pairs.sort((left, right) => left.score - right.score);
}

function getPairSections(modelId, layerId, categoryIds) {
  const pairs = getSortedPairs(modelId, layerId, categoryIds);
  const count = Math.min(30, pairs.length);
  const middleStart = Math.max(0, Math.floor((pairs.length - count) / 2));

  return {
    totalPairs: pairs.length,
    items: [
      {
        title: "Top 30 strongest",
        pairs: pairs.slice(Math.max(0, pairs.length - count)).reverse()
      },
      {
        title: "Middle 30",
        pairs: pairs.slice(middleStart, middleStart + count)
      },
      {
        title: "Lowest 30",
        pairs: pairs.slice(0, count)
      }
    ]
  };
}

function summarizeSubsetMatrix(matrix, categoryIds) {
  const values = [];
  let maxValue = -Infinity;
  let minValue = Infinity;
  let maxPair = [categoryIds[0], categoryIds[1]];
  let minPair = [categoryIds[0], categoryIds[1]];

  for (let row = 0; row < matrix.length; row += 1) {
    for (let column = row + 1; column < matrix.length; column += 1) {
      const value = matrix[row][column];
      values.push(value);

      if (value > maxValue) {
        maxValue = value;
        maxPair = [categoryIds[row], categoryIds[column]];
      }

      if (value < minValue) {
        minValue = value;
        minPair = [categoryIds[row], categoryIds[column]];
      }
    }
  }

  const average = values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;

  return {
    average,
    maxValue,
    minValue,
    maxPair,
    minPair
  };
}

function getProjectionStats(modelId, layerId, categoryIds, focusCategoryId) {
  const matrix = buildMatrix(modelId, layerId, categoryIds);
  const coordinates = modelMap.get(modelId)?.maps?.[layerId] ?? [];
  const focusIndex = categoryIds.indexOf(focusCategoryId);
  const pairValues = [];

  for (let row = 0; row < matrix.length; row += 1) {
    for (let column = row + 1; column < matrix.length; column += 1) {
      pairValues.push(matrix[row][column]);
    }
  }

  const focusValues =
    focusIndex >= 0
      ? matrix[focusIndex].filter((_, index) => index !== focusIndex)
      : [];

  const selectedCoordinates = categoryIds.map((categoryId) => {
    const index = categoryIndexMap.get(categoryId);
    return coordinates[index] ?? [0, 0];
  });

  const xs = selectedCoordinates.map((point) => point[0]);
  const ys = selectedCoordinates.map((point) => point[1]);

  return {
    selectedRange: {
      min: pairValues.length ? Math.min(...pairValues) : 0,
      max: pairValues.length ? Math.max(...pairValues) : 0
    },
    focusRange: {
      min: focusValues.length ? Math.min(...focusValues) : 0,
      max: focusValues.length ? Math.max(...focusValues) : 0
    },
    coordinateRange: {
      minX: xs.length ? Math.min(...xs) : 0,
      maxX: xs.length ? Math.max(...xs) : 0,
      minY: ys.length ? Math.min(...ys) : 0,
      maxY: ys.length ? Math.max(...ys) : 0
    }
  };
}

function buildProjectionSvg(modelId, layerId, categoryIds, focusCategoryId, neighbors, coordinateRange) {
  const model = modelMap.get(modelId);
  const coordinates = model?.maps?.[layerId] ?? [];
  const width = 520;
  const height = 360;
  const padding = 44;
  const focusLabel = categoryMap.get(focusCategoryId)?.label ?? focusCategoryId;

  const points = categoryIds.map((categoryId) => {
    const index = categoryIndexMap.get(categoryId);
    const [x = 0, y = 0] = coordinates[index] ?? [0, 0];
    return {
      id: categoryId,
      label: categoryMap.get(categoryId)?.label ?? categoryId,
      accent: categoryMap.get(categoryId)?.accent ?? "#c75b12",
      x,
      y
    };
  });

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  function scaleX(value) {
    if (Math.abs(maxX - minX) < 1e-6) {
      return width / 2;
    }
    return padding + ((value - minX) / (maxX - minX)) * (width - padding * 2);
  }

  function scaleY(value) {
    if (Math.abs(maxY - minY) < 1e-6) {
      return height / 2;
    }
    return padding + ((value - minY) / (maxY - minY)) * (height - padding * 2);
  }

  const scaledPoints = points.map((point) => ({
    ...point,
    px: scaleX(point.x),
    py: scaleY(point.y)
  }));
  const focusPoint = scaledPoints.find((point) => point.id === focusCategoryId) ?? scaledPoints[0];
  const labeledIds = new Set([focusCategoryId, ...neighbors.map((neighbor) => neighbor.id)]);

  const links = neighbors
    .map((neighbor) => {
      const point = scaledPoints.find((item) => item.id === neighbor.id);
      if (!point || !focusPoint) {
        return "";
      }

      return `
        <line
          x1="${focusPoint.px}"
          y1="${focusPoint.py}"
          x2="${point.px}"
          y2="${point.py}"
          stroke="${interpolateColor(neighbor.score)}"
          stroke-width="${1 + neighbor.score * 4}"
          stroke-opacity="0.6"
        />
      `;
    })
    .join("");

  const dots = scaledPoints
    .map((point) => {
      const isFocus = point.id === focusCategoryId;
      const radius = isFocus ? 10 : labeledIds.has(point.id) ? 6 : 4;
      const label = labeledIds.has(point.id)
        ? `
          <text
            x="${point.px + 10}"
            y="${point.py - 10}"
            fill="#60544e"
            font-size="12"
            font-family="'PT Sans', sans-serif"
          >
            ${escapeHtml(point.label)}
          </text>
        `
        : "";

      return `
        <g>
          <circle
            cx="${point.px}"
            cy="${point.py}"
            r="${radius}"
            fill="${point.accent}"
            stroke="${isFocus ? "#3b2f2a" : "#ffffff"}"
            stroke-width="${isFocus ? 3 : 1.5}"
            opacity="${isFocus ? 1 : 0.88}"
          >
            <title>${escapeHtml(point.label)}</title>
          </circle>
          ${label}
        </g>
      `;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="2D similarity projection for ${escapeHtml(focusLabel)} in ${escapeHtml(layerId)}">
      <defs>
        <linearGradient id="map-bg-${modelId}-${layerId}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fffaf5" />
          <stop offset="100%" stop-color="#f4ece4" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${width}" height="${height}" rx="28" fill="url(#map-bg-${modelId}-${layerId})" />
      <circle cx="${focusPoint?.px ?? width / 2}" cy="${focusPoint?.py ?? height / 2}" r="42" fill="rgba(233,131,0,0.08)" />
      <line x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}" stroke="#e6ddd5" stroke-width="1" />
      <line x1="${width / 2}" y1="${padding}" x2="${width / 2}" y2="${height - padding}" stroke="#e6ddd5" stroke-width="1" />
      <text x="${width - padding}" y="${height / 2 - 8}" fill="#7a726c" font-size="11" font-family="'PT Sans', sans-serif" text-anchor="end">
        x max ${formatSigned(coordinateRange.maxX)}
      </text>
      <text x="${padding}" y="${height / 2 - 8}" fill="#7a726c" font-size="11" font-family="'PT Sans', sans-serif" text-anchor="start">
        x min ${formatSigned(coordinateRange.minX)}
      </text>
      <text x="${width / 2 + 8}" y="${padding + 12}" fill="#7a726c" font-size="11" font-family="'PT Sans', sans-serif">
        y min ${formatSigned(coordinateRange.minY)}
      </text>
      <text x="${width / 2 + 8}" y="${height - padding - 6}" fill="#7a726c" font-size="11" font-family="'PT Sans', sans-serif">
        y max ${formatSigned(coordinateRange.maxY)}
      </text>
      ${links}
      ${dots}
    </svg>
  `;
}

function buildMatrixSvg(key, categoryIds, matrix) {
  const categoryCount = categoryIds.length;
  const cellSize =
    categoryCount > 110 ? 8 : categoryCount > 80 ? 10 : categoryCount > 50 ? 13 : categoryCount > 24 ? 18 : 32;
  const labelFontSize =
    categoryCount > 110 ? 6 : categoryCount > 80 ? 7 : categoryCount > 50 ? 8 : categoryCount > 24 ? 9 : 11;
  const leftMargin = categoryCount > 50 ? 120 : 150;
  const topMargin = categoryCount > 50 ? 120 : 150;
  const innerPadding = categoryCount > 50 ? 1 : 2;
  const showValues = categoryCount <= 10;
  const width = leftMargin + categoryCount * cellSize + 28;
  const height = topMargin + categoryCount * cellSize + 28;

  const labelsTop = categoryIds
    .map((categoryId, index) => {
      const x = leftMargin + index * cellSize + cellSize / 2;
      return `
        <text
          x="${x}"
          y="${topMargin - 16}"
          transform="rotate(-60 ${x} ${topMargin - 16})"
          fill="#60544e"
          font-size="${labelFontSize}"
          font-family="'PT Sans', sans-serif"
          text-anchor="start"
        >
          ${escapeHtml(categoryMap.get(categoryId).label)}
        </text>
      `;
    })
    .join("");

  const labelsLeft = categoryIds
    .map((categoryId, index) => {
      const y = topMargin + index * cellSize + cellSize / 2 + 3;
      return `
        <text
          x="${leftMargin - 10}"
          y="${y}"
          fill="#60544e"
          font-size="${labelFontSize}"
          font-family="'PT Sans', sans-serif"
          text-anchor="end"
        >
          ${escapeHtml(categoryMap.get(categoryId).label)}
        </text>
      `;
    })
    .join("");

  const cells = matrix
    .map((row, rowIndex) =>
      row
        .map((value, columnIndex) => {
          const x = leftMargin + columnIndex * cellSize;
          const y = topMargin + rowIndex * cellSize;
          return `
            <g>
              <rect
                x="${x + innerPadding / 2}"
                y="${y + innerPadding / 2}"
                width="${cellSize - innerPadding}"
                height="${cellSize - innerPadding}"
                fill="${interpolateColor(value)}"
                stroke="#f0e7de"
                stroke-width="0.6"
              />
              ${
                showValues
                  ? `
                    <text
                      x="${x + cellSize / 2}"
                      y="${y + cellSize / 2 + 4}"
                      fill="${value > 0.68 ? "#ffffff" : "#60544e"}"
                      font-size="9"
                      font-family="'PT Sans', sans-serif"
                      text-anchor="middle"
                    >
                      ${value.toFixed(2)}
                    </text>
                  `
                  : ""
              }
            </g>
          `;
        })
        .join("")
    )
    .join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Cosine similarity matrix for ${escapeHtml(key)}">
      <defs>
        <linearGradient id="matrix-backdrop-${key}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fffaf5" />
          <stop offset="100%" stop-color="#f7efe8" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${width}" height="${height}" rx="24" fill="url(#matrix-backdrop-${key})" />
      ${labelsTop}
      ${labelsLeft}
      ${cells}
    </svg>
  `;
}

function interpolateColor(value) {
  const low = [239, 231, 222];
  const mid = [230, 163, 82];
  const high = [199, 91, 18];

  const mix = value < 0.7 ? value / 0.7 : (value - 0.7) / 0.3;
  const source = value < 0.7 ? low : mid;
  const target = value < 0.7 ? mid : high;
  const rgb = source.map((channel, index) => Math.round(channel + (target[index] - channel) * mix));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function formatScore(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value.toFixed(2);
}

function formatSigned(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
}

function formatPair(pairIds) {
  if (!Array.isArray(pairIds) || pairIds.length !== 2) {
    return "n/a";
  }
  return pairIds
    .map((categoryId) => escapeHtml(categoryMap.get(categoryId)?.label ?? categoryId))
    .join(" and ");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

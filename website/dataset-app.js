let datasetData = null;
let categories = [];
let controlsBound = false;

const state = {
  search: "",
  sort: "alpha",
  filter: "all"
};

const searchInput = document.querySelector("#dataset-search");
const sortSelect = document.querySelector("#dataset-sort");
const summaryPanel = document.querySelector("#summary-panel");
const gallery = document.querySelector("#dataset-gallery");
const caption = document.querySelector("#viz-caption");
const datasetCategoryCount = document.querySelector("#dataset-category-count");
const datasetImageCount = document.querySelector("#dataset-image-count");
const datasetPublicCount = document.querySelector("#dataset-public-count");

if (summaryPanel && gallery && caption && searchInput && sortSelect) {
  initializeDatasetPage();
}

async function initializeDatasetPage() {
  renderLoadingState();

  try {
    const response = await fetch("data/cnn-similarity-data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load dataset metadata (${response.status})`);
    }

    datasetData = await response.json();
    categories = datasetData.categories ?? [];
    renderIntroStats();
    bindControls();
    render();
  } catch (error) {
    renderErrorState(error);
  }
}

function renderIntroStats() {
  const dataset = datasetData?.dataset ?? {};

  if (datasetCategoryCount) {
    datasetCategoryCount.textContent = String(dataset.categoryCount ?? categories.length);
  }
  if (datasetImageCount) {
    datasetImageCount.textContent = String(dataset.totalImages ?? 0);
  }
  if (datasetPublicCount) {
    datasetPublicCount.textContent = String(dataset.publicImageCount ?? 0);
  }
}

function bindControls() {
  if (controlsBound) {
    return;
  }

  controlsBound = true;

  searchInput.addEventListener("input", (event) => {
    state.search = event.target.value;
    render();
  });

  sortSelect.addEventListener("change", (event) => {
    state.sort = event.target.value;
    render();
  });

  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      render();
    });
  });
}

function renderLoadingState() {
  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Loading dataset</h3>
        <p>Reading the full category list and public sample thumbnails.</p>
      </article>
    </div>
  `;
  gallery.innerHTML = `<div class="empty-state">Loading category gallery.</div>`;
  caption.textContent = "Loading dataset metadata.";
}

function renderErrorState(error) {
  const message = error instanceof Error ? error.message : String(error);
  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Dataset load failed</h3>
        <p>${escapeHtml(message)}</p>
        <p>Serve the website over HTTP so the browser can load the generated JSON file.</p>
      </article>
    </div>
  `;
  gallery.innerHTML = `
    <div class="empty-state">
      The dataset tab could not load <code>data/cnn-similarity-data.json</code>.
    </div>
  `;
  caption.textContent = "The gallery becomes available after the data file loads.";
}

function getVisibleCategories() {
  const search = state.search.trim().toLowerCase();
  let visible = categories.filter((category) => {
    const matchesSearch =
      !search ||
      category.label.toLowerCase().includes(search) ||
      category.id.toLowerCase().includes(search);

    if (!matchesSearch) {
      return false;
    }

    if (state.filter === "single") {
      return category.sourceCount === 1;
    }

    if (state.filter === "dense") {
      return category.sourceCount >= 10;
    }

    return true;
  });

  if (state.sort === "count-desc") {
    visible = [...visible].sort((left, right) => {
      if (right.sourceCount !== left.sourceCount) {
        return right.sourceCount - left.sourceCount;
      }
      return left.label.localeCompare(right.label);
    });
  } else if (state.sort === "count-asc") {
    visible = [...visible].sort((left, right) => {
      if (left.sourceCount !== right.sourceCount) {
        return left.sourceCount - right.sourceCount;
      }
      return left.label.localeCompare(right.label);
    });
  } else {
    visible = [...visible].sort((left, right) => left.label.localeCompare(right.label));
  }

  return visible;
}

function render() {
  renderSummary();
  renderGallery();
}

function renderSummary() {
  const visible = getVisibleCategories();
  const visibleSourceImages = visible.reduce((sum, category) => sum + (category.sourceCount ?? 0), 0);
  const visiblePublicImages = visible.reduce((sum, category) => sum + (category.publicCount ?? 0), 0);
  const singleImageVisible = visible.filter((category) => category.sourceCount === 1).length;
  const dataset = datasetData?.dataset ?? {};

  summaryPanel.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <h3>Visible slice</h3>
        <p><strong>${visible.length}</strong> categories match the current filters.</p>
        <p><strong>${visibleSourceImages}</strong> source images are represented in those folders.</p>
        <p><strong>${visiblePublicImages}</strong> public thumbnails are shown in the gallery.</p>
      </article>
      <article class="summary-card">
        <h3>Dataset shape</h3>
        <p><strong>${dataset.categoryCount ?? categories.length}</strong> categories in the full archive.</p>
        <p><strong>${dataset.totalImages ?? 0}</strong> source images overall.</p>
        <p><strong>${dataset.singleImageCategoryCount ?? 0}</strong> categories have only one image.</p>
      </article>
      <article class="summary-card">
        <h3>Public release model</h3>
        <p>The site shows only copied thumbnails, not the full dataset files.</p>
        <p><strong>${singleImageVisible}</strong> currently visible categories have exactly one image.</p>
        <p><a href="cnn.html">Open the CNN tab</a> to inspect similarities built from all source images.</p>
      </article>
    </div>
  `;
}

function renderGallery() {
  const visible = getVisibleCategories();

  if (!visible.length) {
    gallery.innerHTML = `
      <div class="empty-state">
        No categories match the current search and filter settings.
      </div>
    `;
    caption.textContent = "Adjust the search field or filter buttons to reveal categories.";
    return;
  }

  caption.textContent = `Showing ${visible.length} categories from the public subset.`;
  gallery.innerHTML = visible
    .map(
      (category) => `
        <article class="dataset-card" style="--chip-accent: ${category.accent};">
          <div class="dataset-card-header">
            <div>
              <h3>${escapeHtml(category.label)}</h3>
              <p class="sample-category-meta">${escapeHtml(category.note)}</p>
            </div>
          </div>
          <div class="dataset-thumb-grid">
            ${category.samples
              .map(
                (samplePath, index) => `
                  <img
                    src="${samplePath}"
                    alt="${escapeHtml(category.label)} sample ${index + 1}"
                    loading="lazy"
                  />
                `
              )
              .join("")}
          </div>
        </article>
      `
    )
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

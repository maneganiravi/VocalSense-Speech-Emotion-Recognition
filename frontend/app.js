const EMOTION_CODES = {
  "01": "neutral",
  "02": "calm",
  "03": "happy",
  "04": "sad",
  "05": "angry",
  "06": "fearful",
  "07": "disgust",
  "08": "surprised"
};

const state = {
  file: null,
  objectUrl: null,
  health: null,
  modelInfo: null,
  metrics: null,
  dataset: null,
  dashboard: null,
  prediction: null,
  history: [],
  sampleOptions: []
};
const $ = (selector) => document.querySelector(selector);

function formatTimestamp(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function formatPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return `${(value * 100).toFixed(1)}%`;
}

function supportedLabels() {
  return state.modelInfo?.class_order || state.metrics?.class_order || [];
}

function updatePredictButton() {
  const button = $("#predict-button");
  if (!button) return;
  button.disabled = !state.file || !state.health?.models_ready || !state.modelInfo?.ready;
}

function renderModelStatus() {
  const ready = Boolean(state.health?.models_ready && state.modelInfo?.ready);
  const cnnReady = Boolean(state.modelInfo?.models?.cnn?.ready_for_inference);
  const bilstmReady = Boolean(state.modelInfo?.models?.cnn_bilstm?.ready_for_inference);
  const detail = state.modelInfo?.detail || state.health?.detail || "Model status unavailable.";
  const statusIndicator = $("#status-indicator");
  if (statusIndicator) {
    statusIndicator.classList.toggle("online", ready);
    statusIndicator.classList.toggle("offline", !ready);
  }
  const statusText = $("#status-text");
  if (statusText) {
    statusText.textContent = state.health
      ? ready ? "Models ready" : "Models unavailable"
      : "API unavailable";
    statusText.title = detail;
  }
  const aboutModel = $("#about-model");
  if (aboutModel) aboutModel.textContent = ready ? "CNN + CNN-BiLSTM ready" : detail;

  const pill = $("#dashboard-status-pill");
  if (pill) {
    pill.textContent = ready ? "Models ready" : "Models unavailable";
    pill.classList.toggle("online", ready);
    pill.classList.toggle("offline", !ready);
  }

  const overviewBackend = $("#overview-backend-status");
  const overviewCnn = $("#overview-cnn-status");
  const overviewBilstm = $("#overview-bilstm-status");
  if (overviewBackend) overviewBackend.textContent = state.health ? "Connected" : "Offline";
  if (overviewCnn) overviewCnn.textContent = cnnReady ? "Ready" : "Unavailable";
  if (overviewBilstm) overviewBilstm.textContent = bilstmReady ? "Ready" : "Unavailable";

  const modelStatus = $("#dashboard-model-status");
  if (modelStatus) modelStatus.textContent = ready ? "Ready" : "Offline";
  const modelDetail = $("#dashboard-model-detail");
  if (modelDetail) {
    modelDetail.textContent = state.dashboard?.model_dataset_detail || detail;
    modelDetail.title = detail;
  }
  updatePredictButton();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
  return payload;
}

function showError(message) {
  const error = $("#error-message");
  error.textContent = message;
  error.classList.remove("hidden");
}

function clearError() {
  $("#error-message").classList.add("hidden");
}

function setFile(file) {
  clearError();
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav")) {
    showError("Please choose a WAV (.wav) audio file.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("The audio file must be 20 MB or smaller.");
    return;
  }
  if (file.size === 0) {
    showError("The selected WAV file is empty.");
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.prediction = null;
  $("#empty-result").classList.remove("hidden");
  $("#prediction-results").classList.add("hidden");
  $("#loading-state").classList.add("hidden");
  state.objectUrl = URL.createObjectURL(file);
  $("#file-name").textContent = file.name;
  $("#file-details").textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · WAV audio`;
  $("#selected-file").classList.remove("hidden");
  $("#audio-player").src = state.objectUrl;
  $("#audio-player").classList.remove("hidden");
  updatePredictButton();
}

function removeFile() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  state.file = null;
  state.prediction = null;
  $("#audio-input").value = "";
  $("#selected-file").classList.add("hidden");
  $("#audio-player").classList.add("hidden");
  $("#audio-player").removeAttribute("src");
  updatePredictButton();
  clearError();
}

function renderPrediction(prediction, duration) {
  state.prediction = prediction;
  const selectedModel = prediction.selected_model || $("#analysis-model-select").value || "ensemble";
  const selectedResult = prediction.selected_prediction || prediction[selectedModel] || prediction.ensemble;
  $("#prediction-model").value = selectedModel;
  $("#empty-result").classList.add("hidden");
  $("#loading-state").classList.add("hidden");
  $("#prediction-results").classList.remove("hidden");
  $("#winner-label").textContent = `${selectedModel.replaceAll("_", " ").toUpperCase()} PREDICTION`;
  $("#winner-emotion").textContent = selectedResult.emotion;
  $("#prediction-confidence").textContent = `${formatPercent(selectedResult.confidence)} confidence`;
  $("#cnn-emotion").textContent = prediction.cnn.emotion;
  $("#bilstm-emotion").textContent = prediction.cnn_bilstm.emotion;
  $("#ensemble-emotion").textContent = prediction.ensemble.emotion;
  $("#prediction-time").textContent = `Audio duration: ${duration.toFixed(1)} sec`;
  renderPredictionProbabilities();
  $("#analyze-another").classList.remove("hidden");
  loadHistory();
}

function renderPredictionProbabilities() {
  if (!state.prediction) return;
  const chart = $("#probability-chart");
  chart.replaceChildren();
  const modelName = $("#prediction-model").value || "ensemble";
  const result = state.prediction[modelName];
  const probabilities = result?.probabilities || {};
  if (result) {
    $("#winner-label").textContent = `${modelName.replaceAll("_", " ").toUpperCase()} PREDICTION`;
    $("#winner-emotion").textContent = result.emotion;
    $("#prediction-confidence").textContent = `${formatPercent(result.confidence)} confidence`;
    $("#analysis-model-select").value = modelName;
  }
  const entries = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);
  for (const [label, probability] of entries) {
    const row = document.createElement("div");
    row.className = "prob-row";
    const name = document.createElement("span");
    name.className = "prob-label";
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "prob-track";
    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "prob-value";
    value.textContent = `${(probability * 100).toFixed(1)}%`;
    row.append(name, track, value);
    chart.append(row);
  }
}

async function predict() {
  if (!state.file) return;
  clearError();
  $("#empty-result").classList.add("hidden");
  $("#prediction-results").classList.add("hidden");
  $("#loading-state").classList.remove("hidden");
  $("#predict-button").disabled = true;
  $(".button-text").textContent = "Analyzing…";
  $("#predict-button").setAttribute("aria-busy", "true");
  try {
    const data = new FormData();
    data.append("file", state.file);
    data.append("selected_model", $("#analysis-model-select").value || "ensemble");
    const result = await api("/api/predict", { method: "POST", body: data });
    renderPrediction(result, result.duration_seconds);
  } catch (error) {
    $("#loading-state").classList.add("hidden");
    $("#empty-result").classList.remove("hidden");
    showError(error.message);
  } finally {
    $("#predict-button").removeAttribute("aria-busy");
    updatePredictButton();
    $(".button-text").textContent = "Analyze emotion";
  }
}

function renderDashboardOverview() {
  const dashboard = state.dashboard || {};
  const ready = Boolean(state.health?.models_ready && state.modelInfo?.ready);
  const labels = supportedLabels();
  const totalPredictions = dashboard.total_predictions ?? state.history.length;
  const modelCount = state.modelInfo?.loaded_model_count
    ?? state.health?.loaded_model_count
    ?? dashboard.loaded_model_count
    ?? dashboard.model_count;
  const datasetAvailable = state.dataset?.dataset_available
    ?? state.dataset?.configured
    ?? dashboard.dataset_available
    ?? dashboard.dataset_configured;

  const totalPredictionsEl = $("#dashboard-total-predictions");
  if (totalPredictionsEl) totalPredictionsEl.textContent = Number.isFinite(totalPredictions) ? Number(totalPredictions).toLocaleString() : "Not available";
  const totalMeta = $("#dashboard-total-predictions-meta");
  if (totalMeta) totalMeta.textContent = dashboard.last_prediction ? `Last: ${formatTimestamp(dashboard.last_prediction)}` : "Latest history";

  const modelsLoadedEl = $("#dashboard-models-loaded");
  if (modelsLoadedEl) modelsLoadedEl.textContent = Number.isInteger(modelCount) ? String(modelCount) : "Not available";
  const classCount = $("#dashboard-class-count");
  if (classCount) classCount.textContent = labels.length ? String(labels.length) : "—";
  const datasetStatus = $("#dashboard-dataset-status");
  if (datasetStatus) {
    datasetStatus.textContent = datasetAvailable === true
      ? "Available"
      : datasetAvailable === false
        ? "Not available"
        : "Status unavailable";
  }
  const datasetDetail = $("#dashboard-dataset-detail");
  if (datasetDetail) {
    const count = state.dataset?.valid_audio_count ?? state.dataset?.valid_audio_files ?? dashboard.valid_audio_count ?? dashboard.dataset_files;
    const detail = state.dataset?.error || dashboard.dataset_error || state.dataset?.detail || state.dashboardError;
    if (datasetAvailable === true && Number.isInteger(count)) {
      const discovered = state.dataset?.total_wav_files ?? dashboard.total_wav_files;
      datasetDetail.textContent = Number.isInteger(discovered)
        ? `${count.toLocaleString()} valid clips · ${discovered.toLocaleString()} WAVs discovered`
        : `${count.toLocaleString()} valid WAV files`;
    } else if (detail) {
      datasetDetail.textContent = detail;
    } else if (datasetAvailable === false) {
      datasetDetail.textContent = state.dataset?.directory_exists === false ? "Configured directory not found" : "No valid RAVDESS speech files found";
    } else {
      datasetDetail.textContent = "Dataset status could not be loaded";
    }
    datasetDetail.title = state.dataset?.dataset_path || dashboard.dataset_path || detail || "";
  }
  const modelReadinessDetail = $("#dashboard-model-detail");
  if (modelReadinessDetail && dashboard.model_dataset_detail) {
    modelReadinessDetail.textContent = dashboard.model_dataset_detail;
  }

  const overviewUpdated = $("#overview-last-updated");
  if (overviewUpdated) overviewUpdated.textContent = dashboard.last_updated ? formatTimestamp(dashboard.last_updated) : formatTimestamp(state.history?.[0]?.created_at || null);

  const backendStatus = $("#overview-backend-status");
  if (backendStatus) backendStatus.textContent = state.health ? (ready ? "Connected" : "Partial") : "Offline";

  const cnnStatus = $("#overview-cnn-status");
  if (cnnStatus) cnnStatus.textContent = ready ? "Ready" : "Offline";
  const bilstmStatus = $("#overview-bilstm-status");
  if (bilstmStatus) bilstmStatus.textContent = ready ? "Ready" : "Offline";
}

function renderModelSummary() {
  const cards = $("#model-summary-cards");
  if (!cards) return;
  const metrics = state.metrics?.models || {};
  const entries = [
    ["CNN", metrics.cnn?.results?.validation ?? null, metrics.cnn?.results?.test ?? null],
    ["CNN-BiLSTM", metrics.cnn_bilstm?.results?.validation ?? null, metrics.cnn_bilstm?.results?.test ?? null],
    ["Ensemble", metrics.ensemble?.results?.validation ?? null, metrics.ensemble?.results?.test ?? null],
  ];
  cards.replaceChildren(...entries.map(([name, validation, test]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    const title = document.createElement("span");
    title.textContent = name;
    const accuracy = document.createElement("strong");
    accuracy.textContent = `Test accuracy ${formatPercent(test?.accuracy)}`;
    const details = document.createElement("small");
    details.textContent = `Validation accuracy ${formatPercent(validation?.accuracy)} · Test macro F1 ${formatPercent(test?.macro_f1)} · Test weighted F1 ${formatPercent(test?.weighted_f1)}`;
    card.append(title, accuracy, details);
    return card;
  }));
}

function renderRecentPredictions() {
  const list = $("#recent-predictions-list");
  if (!list) return;
  const items = (state.history || []).slice(0, 4);
  if (!items.length) {
    list.innerHTML = '<div class="empty-mini-state">No recent predictions yet.</div>';
    return;
  }

  list.replaceChildren(...items.map((item) => {
    const row = document.createElement("div");
    row.className = "recent-item";
    const info = document.createElement("div");
    const filename = document.createElement("strong");
    filename.textContent = item.filename;
    const detail = document.createElement("small");
    detail.textContent = `${item.emotion} · ${(Number(item.duration_seconds) || 0).toFixed(1)}s · ${(item.model_used || "ensemble").replaceAll("_", " ")}`;
    info.append(filename, detail);
    const confidence = document.createElement("span");
    confidence.textContent = typeof item.confidence === "number" ? formatPercent(item.confidence) : "Not available";
    const details = document.createElement("button");
    details.type = "button";
    details.className = "text-button";
    details.textContent = "Details";
    details.addEventListener("click", () => {
      const historyNav = document.querySelector('.nav-item[data-page="history"]');
      if (historyNav) historyNav.click();
      showHistoryDetail(item.id);
    });
    row.append(info, confidence, details);
    return row;
  }));
}

function renderEmotionDistribution() {
  const chart = $("#emotion-distribution-chart");
  const empty = $("#emotion-distribution-empty");
  if (!chart) return;
  const totals = {};
  (state.history || []).forEach((item) => {
    const emotion = item.emotion || "unknown";
    totals[emotion] = (totals[emotion] || 0) + 1;
  });

  if (!Object.keys(totals).length) {
    chart.innerHTML = "";
    if (empty) empty.classList.remove("hidden");
    return;
  }

  if (empty) empty.classList.add("hidden");
  const entries = Object.entries(totals);
  const maxValue = Math.max(...entries.map(([, count]) => count), 1);
  chart.replaceChildren(...entries.map(([label, count]) => {
    const row = document.createElement("div");
    row.className = "distribution-row";
    const labelRow = document.createElement("div");
    labelRow.className = "distribution-label-row";
    const name = document.createElement("span");
    name.textContent = label;
    const total = document.createElement("strong");
    total.textContent = String(count);
    labelRow.append(name, total);
    const bar = document.createElement("div");
    bar.className = "distribution-bar";
    const fill = document.createElement("span");
    fill.style.width = `${(count / maxValue) * 100}%`;
    bar.append(fill);
    row.append(labelRow, bar);
    return row;
  }));
}

function renderSampleOptions() {
  const select = $("#sample-audio-select");
  if (!select) return;
  select.innerHTML = '<option value="">Pick from dataset</option>';
  const sampleOptions = state.dataset?.samples || state.sampleOptions || [];
  sampleOptions.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.filename;
    option.textContent = `${sample.emotion} · ${sample.filename}`;
    select.appendChild(option);
  });
}

async function loadDashboardSummary() {
  try {
    state.dashboard = await api("/api/dashboard");
    state.dashboardError = null;
  } catch (error) {
    state.dashboard = null;
    state.dashboardError = `Dashboard summary unavailable: ${error.message}`;
  }
  renderDashboardOverview();
}

function updateModelAvailability() {
  const ready = Boolean(state.health?.models_ready && state.modelInfo?.ready);
  const select = $("#analysis-model-select");
  if (!select) return;
  Array.from(select.options).forEach((option) => {
    const shouldDisable = !ready && option.value !== "ensemble";
    option.disabled = shouldDisable;
  });
  if (!ready) {
    select.value = "ensemble";
  }
}

function renderHistory(items) {
  state.history = items;
  const container = $("#history-table");
  const query = ($("#history-search")?.value || "").trim().toLowerCase();
  const filtered = items.filter((item) =>
    `${item.filename} ${item.emotion}`.toLowerCase().includes(query)
  );
  if (!filtered.length) {
    container.innerHTML = items.length
      ? '<div class="empty-table">No saved predictions match your search.</div>'
      : '<div class="empty-table">No predictions yet. Analyze a WAV file to start your history.</div>';
    return;
  }
  const table = document.createElement("table");
  table.className = "data-table";
  table.innerHTML = "<thead><tr><th>FILE</th><th>PREDICTED EMOTION</th><th>MODEL</th><th>CONFIDENCE</th><th>DURATION</th><th>DATE</th><th></th></tr></thead>";
  const body = document.createElement("tbody");
  for (const item of filtered) {
    const row = document.createElement("tr");
    for (const value of [
      item.filename,
      item.emotion,
      (item.model_used || "ensemble").replaceAll("_", " "),
      typeof item.confidence === "number" ? formatPercent(item.confidence) : "Not available",
      `${Number(item.duration_seconds).toFixed(1)} sec`,
      new Date(item.created_at).toLocaleString()
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const actionCell = document.createElement("td");
    const inspectButton = document.createElement("button");
    inspectButton.className = "text-button";
    inspectButton.type = "button";
    inspectButton.textContent = "Details";
    inspectButton.addEventListener("click", () => showHistoryDetail(item.id));
    actionCell.append(inspectButton);
    row.append(actionCell);
    body.append(row);
  }
  table.append(body);
  container.replaceChildren(table);
}

async function loadHistory() {
  try {
    const items = await api("/api/history?limit=100");
    renderHistory(items);
    const historyCount = $("#dashboard-history-count");
    if (historyCount) historyCount.textContent = String(items.length);
    renderRecentPredictions();
    renderEmotionDistribution();
    await loadDashboardSummary();
    renderDashboardOverview();
  } catch (error) {
    $("#history-table").textContent = `Could not load prediction history: ${error.message}`;
  }
}

async function showHistoryDetail(id) {
  try {
    const detail = await api(`/api/history/${encodeURIComponent(id)}`);
    $("#history-detail-title").textContent = detail.filename;
    const content = $("#history-detail-content");
    content.replaceChildren();
    const summary = document.createElement("p");
    const confidence = typeof detail.confidence === "number"
      ? `${(detail.confidence * 100).toFixed(1)}%`
      : "Not available";
    summary.textContent = `Prediction: ${detail.emotion} · Model: ${(detail.model_used || "ensemble").replaceAll("_", " ")} · Confidence ${confidence} · Duration ${Number(detail.duration_seconds).toFixed(1)} sec`;
    content.append(summary);
    const predictions = detail.probabilities || {};
    for (const [name, result] of Object.entries(predictions)) {
      const section = document.createElement("section");
      section.className = "history-model-detail";
      const heading = document.createElement("h3");
      heading.textContent = `${name.replaceAll("_", " ")} · ${result.emotion}`;
      section.append(heading);
      const probabilities = document.createElement("p");
      probabilities.textContent = Object.entries(result.probabilities || {})
        .map(([label, value]) => `${label}: ${(value * 100).toFixed(1)}%`)
        .join(" · ");
      section.append(probabilities);
      content.append(section);
    }
    $("#history-detail").classList.remove("hidden");
  } catch (error) {
    $("#history-table").textContent = `Could not load prediction details: ${error.message}`;
  }
}

function formatPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function getMetricResult(modelName, split = "validation") {
  const model = state.metrics?.models?.[modelName];
  if (!model || !model.results) return null;
  return model.results[split] || model.results.validation || model.results.test || null;
}

function renderPerformance(info) {
  const notice = $("#performance-notice");
  const content = $("#performance-content");
  const metrics = info?.metrics || state.metrics || null;
  if (!info?.ready || !metrics?.models) {
    notice.textContent = info?.detail || state.metricsError || "Evaluation metrics are not available.";
    notice.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }
  notice.classList.add("hidden");
  content.classList.remove("hidden");
  const visibleModels = ["cnn", "cnn_bilstm", "ensemble"];
  const selectedSplit = $("#performance-split").value || "validation";
  const metricNames = [
    ["Accuracy", "accuracy"],
    ["Macro F1", "macro_f1"],
    ["Weighted F1", "weighted_f1"],
    ["Macro precision", "macro_precision"],
    ["Macro recall", "macro_recall"]
  ];
  const table = document.createElement("table");
  table.className = "data-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Model", ...metricNames.map(([label]) => label)].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headRow.append(cell);
  });
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const modelName of visibleModels) {
    const result = metrics.models?.[modelName]?.results?.[selectedSplit];
    if (!result) continue;
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.textContent = modelName.replaceAll("_", " ").toUpperCase();
    row.append(name);
    for (const [, key] of metricNames) {
      const cell = document.createElement("td");
      cell.textContent = formatPercent(result[key]);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
  $("#metrics-table").replaceChildren(table);

  const metric = $("#performance-metric").value || "accuracy";
  const ranked = visibleModels
    .map((name) => ({ name, score: metrics.models?.[name]?.results?.[selectedSplit]?.[metric] }))
    .filter((entry) => typeof entry.score === "number" && Number.isFinite(entry.score))
    .sort((left, right) => right.score - left.score);
  $("#performance-best-model").textContent = ranked.length
    ? `Best on ${selectedSplit} ${metric.replaceAll("_", " ")}: ${ranked[0].name.replaceAll("_", " ").toUpperCase()} (${formatPercent(ranked[0].score)})`
    : "Best model: Not available";

  const selectedModel = $("#confusion-model").value || "ensemble";
  const result = metrics.models?.[selectedModel]?.results?.[selectedSplit];
  const labels = metrics.class_order || info.class_order || [];
  const matrix = document.createElement("table");
  matrix.className = "data-table confusion-matrix";
  const matrixHead = document.createElement("thead");
  const matrixHeadRow = document.createElement("tr");
  ["Actual \\ Predicted", ...labels].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    matrixHeadRow.append(cell);
  });
  matrixHead.append(matrixHeadRow);
  const matrixBody = document.createElement("tbody");
  (result?.confusion_matrix || []).forEach((values, rowIndex) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.textContent = labels[rowIndex] || `Class ${rowIndex + 1}`;
    row.append(label);
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    matrixBody.append(row);
  });
  matrix.append(matrixHead, matrixBody);
  $("#confusion-table").replaceChildren(matrix);
}

function renderDataset(dataset) {
  state.dataset = dataset;
  state.sampleOptions = dataset?.samples || [];
  const aboutDataset = $("#about-dataset");
  if (aboutDataset) {
    aboutDataset.textContent = dataset?.dataset_available
      ? `${(dataset.valid_audio_count ?? dataset.valid_audio_files).toLocaleString()} files · ${dataset.actor_count} actors`
      : dataset?.error || "Dataset status unavailable.";
  }
  renderSampleOptions();
  const classCount = $("#dashboard-class-count");
  if (classCount) classCount.textContent = supportedLabels().length || "Not available";
  const datasetStatus = $("#dashboard-dataset-status");
  if (datasetStatus) {
    datasetStatus.textContent = dataset?.dataset_available
      ? "Available"
      : dataset?.dataset_available === false
        ? "Not available"
        : "Status unavailable";
  }
  const datasetDetail = $("#dashboard-dataset-detail");
  if (datasetDetail) {
    datasetDetail.textContent = dataset?.dataset_available
      ? `${(dataset.valid_audio_count ?? dataset.valid_audio_files ?? 0).toLocaleString()} valid clips · ${Number.isInteger(dataset.total_wav_files) ? `${dataset.total_wav_files.toLocaleString()} WAVs discovered` : "discovered count unavailable"}`
      : dataset?.error || "Dataset status unavailable.";
    datasetDetail.title = dataset?.dataset_path || dataset?.error || "";
  }
}

function renderNotebookDataset(metrics) {
  const distribution = metrics.dataset?.emotion_distribution || {};
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);
  const summary = $("#notebook-dataset-summary");
  summary.replaceChildren(...[
    ["TOTAL WAV FILES", metrics.dataset?.total_wav_files ?? "Not available"],
    ["SELECTED CLIPS", metrics.dataset?.selected_clips ?? "Not available"],
    ["UNIQUE ACTORS", new Set(Object.values(metrics.splits || {}).flatMap((split) => split.actors || [])).size || "Not available"],
    ["EMOTION CLASSES", metrics.class_order?.length ?? "Not available"],
  ].map(([label, value]) => { const card = document.createElement("div"); card.className = "metric-card"; const title = document.createElement("span"); title.textContent = label; const number = document.createElement("strong"); number.textContent = value; card.append(title, number); return card; }));
  const datasetNote = $("#notebook-dataset-note");
  if (datasetNote) {
    const live = state.dataset;
    datasetNote.textContent = live?.dataset_available
      ? `Saved training snapshot: ${metrics.dataset?.total_wav_files ?? "not available"} WAVs discovered and ${metrics.dataset?.selected_clips ?? "not available"} clips selected. Current dataset scan: ${live.total_wav_files ?? "not available"} WAVs discovered and ${live.valid_audio_count ?? live.valid_audio_files ?? "not available"} valid clips. These counts describe different scan/training snapshots; training split settings are unchanged.`
      : "Counts shown here come from saved training metrics. Current dataset status is unavailable.";
  }
  if (window.Plotly && labels.length) {
    const trace = { type: "bar", x: labels, y: values, marker: { color: "#2ec4c7" }, text: values.map(String), textposition: "outside" };
    Plotly.newPlot("notebook-dataset-chart", [trace], { margin: { t: 20, r: 10, b: 50, l: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dfeaf2" }, xaxis: { title: "Emotion" }, yaxis: { title: "Count" } }, { responsive: true, displayModeBar: false });
  } else {
    $("#notebook-dataset-chart").textContent = "Dataset chart unavailable.";
  }
  const records = metrics.dataset?.sample_records || [];
  const table = document.createElement("table");
  table.className = "data-table";
  const header = ["File", "Emotion", "Actor", "Path"];
  const thead = document.createElement("thead");
  const row = document.createElement("tr");
  header.forEach((label) => { const th = document.createElement("th"); th.textContent = label; row.appendChild(th); });
  thead.appendChild(row);
  const tbody = document.createElement("tbody");
  records.slice(0, 10).forEach((record) => {
    const tr = document.createElement("tr");
    [record.filename, record.emotion, `Actor_${String(record.actor).padStart(2, "0")}`, record.path].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  $("#notebook-samples-table").replaceChildren(table);
}

function renderNotebookFeatures(metrics) {
  const summary = $("#notebook-feature-summary");
  const featureShape = metrics.features?.shape || [];
  const temporalShape = metrics.features?.temporal_shape || [];
  summary.replaceChildren(...[
    ["INPUT SHAPE", featureShape.length ? featureShape.join(" × ") : "Not available"],
    ["TEMPORAL SHAPE", temporalShape.length ? temporalShape.join(" × ") : "Not available"],
    ["SUMMARY FEATURES", featureShape[1] ?? "Not available"],
    ["SCALER TYPE", metrics.features?.scaler?.type || "Not available"],
  ].map(([label, value]) => { const card = document.createElement("div"); card.className = "metric-card"; const title = document.createElement("span"); title.textContent = label; const number = document.createElement("strong"); number.textContent = value; card.append(title, number); return card; }));
  const description = metrics.features?.summary_description || {};
  $("#notebook-feature-description").textContent = description.mfcc_mean_std
    ? `CNN and CNN-BiLSTM summary: mean and standard deviation of ${description.mfcc_mean_std / 2} MFCC coefficients, ZCR, RMS, and chroma at ${description.sample_rate} Hz. Temporal models use interpolated 64 × 54 frame sequences.`
    : "Feature configuration is not available in the saved metrics.";
  const scalerInfo = $("#notebook-scaler-info");
  const scaler = metrics.features?.scaler;
  scalerInfo.textContent = scaler
    ? `Summary StandardScaler: ${scaler.feature_count} features, fit on ${scaler.fit_samples} training samples. ${scaler.temporal_type}; ${scaler.temporal_fit_frames} training frames.`
    : "Scaler details are not available in the saved metrics.";
}

function renderNotebookSplit(metrics) {
  const table = document.createElement("table");
  table.className = "data-table";
  const rows = Object.entries(metrics.splits || {});
  const header = ["Split", "Samples", "Actors", "Emotion counts"];
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  header.forEach((label) => { const th = document.createElement("th"); th.textContent = label; headRow.appendChild(th); });
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  rows.forEach(([splitName, split]) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td"); name.textContent = splitName;
    const samples = document.createElement("td"); samples.textContent = split.samples ?? 0;
    const actors = document.createElement("td"); actors.textContent = (split.actors || []).join(", ") || "—";
    const classes = document.createElement("td"); classes.textContent = Object.entries(split.class_distribution || {}).map(([emotion, count]) => `${emotion}: ${count}`).join(" · ");
    tr.append(name, samples, actors, classes);
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  $("#notebook-splits-table").replaceChildren(table);
}

function renderModelStage(modelName, containerId, architectureId, chartId) {
  const metrics = state.metrics;
  const model = metrics.models?.[modelName];
  if (!model) return;
  const summary = document.getElementById(containerId);
  const resultValidation = model.results?.validation || model.results?.test || {};
  const resultTest = model.results?.test || model.results?.validation || {};
  summary.replaceChildren(...[
    ["VALIDATION ACCURACY", formatPercent(resultValidation.accuracy)],
    ["VALIDATION MACRO F1", formatPercent(resultValidation.macro_f1)],
    ["TEST ACCURACY", formatPercent(resultTest.accuracy)],
    ["PARAMETERS", Number(model.parameter_count || 0).toLocaleString()],
  ].map(([label, value]) => { const card = document.createElement("div"); card.className = "metric-card"; const title = document.createElement("span"); title.textContent = label; const number = document.createElement("strong"); number.textContent = value; card.append(title, number); return card; }));
  const architectureText = document.getElementById(architectureId);
  const bestEpoch = model.history?.best_epoch ?? "—";
  architectureText.textContent = `${modelName.toUpperCase().replaceAll("_", " ")} | best epoch: ${bestEpoch}\nparameter_count: ${Number(model.parameter_count || 0).toLocaleString()}\nloss: ${model.history?.loss?.length ? model.history.loss[0].toFixed(4) : "—"}\nvalidation_loss: ${model.history?.val_loss?.length ? model.history.val_loss[0].toFixed(4) : "—"}`;
  const history = model.history || {};
  const epochs = Array.from({ length: Math.max(1, history.accuracy?.length || 1) }, (_, index) => index + 1);
  if (window.Plotly && (history.accuracy || history.val_accuracy || history.loss || history.val_loss)) {
    const traces = [];
    if (history.accuracy) traces.push({ x: epochs, y: history.accuracy, mode: "lines", name: "accuracy", line: { color: "#5eead4" } });
    if (history.val_accuracy) traces.push({ x: epochs, y: history.val_accuracy, mode: "lines", name: "val_accuracy", line: { color: "#7dd3fc" } });
    if (history.loss) traces.push({ x: epochs, y: history.loss, mode: "lines", name: "loss", yaxis: "y2", line: { color: "#f7b267" } });
    if (history.val_loss) traces.push({ x: epochs, y: history.val_loss, mode: "lines", name: "val_loss", yaxis: "y2", line: { color: "#ef476f" } });
    Plotly.newPlot(chartId, traces, { margin: { t: 20, r: 20, b: 50, l: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dfeaf2" }, xaxis: { title: "Epoch" }, yaxis: { title: "Accuracy" }, yaxis2: { title: "Loss", overlaying: "y", side: "right", showgrid: false }, legend: { orientation: "h" } }, { responsive: true, displayModeBar: false });
  } else {
    document.getElementById(chartId).textContent = "Training curves unavailable.";
  }
}

function renderNotebookComparison(metrics) {
  const table = document.createElement("table");
  table.className = "data-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Model", "Validation Accuracy", "Validation Macro F1", "Validation Weighted F1", "Test Accuracy", "Test Macro F1", "Test Weighted F1"].forEach((label) => { const th = document.createElement("th"); th.textContent = label; headRow.appendChild(th); });
  head.appendChild(headRow);
  const body = document.createElement("tbody");
  Object.entries(metrics.models || {}).forEach(([name, model]) => {
    const validation = model.results?.validation || model.results?.test || {};
    const test = model.results?.test || model.results?.validation || {};
    const row = document.createElement("tr");
    const modelCell = document.createElement("td"); modelCell.textContent = name.replaceAll("_", " ");
    const valAcc = document.createElement("td"); valAcc.textContent = formatPercent(validation.accuracy);
    const valMacro = document.createElement("td"); valMacro.textContent = formatPercent(validation.macro_f1);
    const valWeighted = document.createElement("td"); valWeighted.textContent = formatPercent(validation.weighted_f1);
    const testAcc = document.createElement("td"); testAcc.textContent = formatPercent(test.accuracy);
    const testMacro = document.createElement("td"); testMacro.textContent = formatPercent(test.macro_f1);
    const testWeighted = document.createElement("td"); testWeighted.textContent = formatPercent(test.weighted_f1);
    row.append(modelCell, valAcc, valMacro, valWeighted, testAcc, testMacro, testWeighted);
    body.appendChild(row);
  });
  table.append(head, body);
  $("#notebook-comparison-table").replaceChildren(table);
}

function renderClassificationReport(metrics) {
  const selectModel = $("#notebook-report-model");
  const splitSelect = $("#notebook-report-split");
  const modelName = selectModel.value || "cnn";
  const split = splitSelect.value || "validation";
  const model = metrics.models?.[modelName];
  const results = model?.results?.[split];
  const table = document.createElement("table");
  table.className = "data-table";
  const headers = ["Emotion", "Precision", "Recall", "F1", "Support"];
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((label) => { const th = document.createElement("th"); th.textContent = label; headRow.appendChild(th); });
  head.appendChild(headRow);
  const body = document.createElement("tbody");
  Object.entries(results?.per_class || {}).forEach(([emotion, details]) => {
    const row = document.createElement("tr");
    [emotion, formatPercent(details.precision), formatPercent(details.recall), formatPercent(details.f1), details.support].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; row.appendChild(td);
    });
    body.appendChild(row);
  });
  table.append(head, body);
  $("#notebook-classification-table").replaceChildren(table);
  if (window.Plotly && results?.confusion_matrix) {
    const z = results.confusion_matrix;
    Plotly.newPlot("notebook-confusion-chart", [{ type: "heatmap", z, x: metrics.class_order || [], y: metrics.class_order || [], colorscale: "Teal" }], { margin: { t: 20, r: 20, b: 80, l: 80 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dfeaf2" }, xaxis: { title: "Predicted" }, yaxis: { title: "Actual" } }, { responsive: true, displayModeBar: false });
  } else {
    $("#notebook-confusion-chart").textContent = "Confusion matrix unavailable.";
  }
}

function renderExperimentCurve(modelName, containerId) {
  const model = state.metrics?.models?.[modelName];
  const history = model?.history || {};
  const epochs = Array.from({ length: Math.max(1, history.accuracy?.length || 1) }, (_, index) => index + 1);
  const traces = [];
  if (history.accuracy) traces.push({ x: epochs, y: history.accuracy, mode: "lines", name: "accuracy", line: { color: "#5eead4" } });
  if (history.val_accuracy) traces.push({ x: epochs, y: history.val_accuracy, mode: "lines", name: "val_accuracy", line: { color: "#7dd3fc" } });
  if (history.loss) traces.push({ x: epochs, y: history.loss, mode: "lines", name: "loss", yaxis: "y2", line: { color: "#f7b267" } });
  if (history.val_loss) traces.push({ x: epochs, y: history.val_loss, mode: "lines", name: "val_loss", yaxis: "y2", line: { color: "#ef476f" } });
  if (window.Plotly && traces.length) {
    Plotly.newPlot(containerId, traces, { margin: { t: 20, r: 20, b: 50, l: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dfeaf2" }, xaxis: { title: "Epoch" }, yaxis: { title: "Accuracy" }, yaxis2: { title: "Loss", overlaying: "y", side: "right", showgrid: false }, legend: { orientation: "h" } }, { responsive: true, displayModeBar: false });
  } else {
    document.getElementById(containerId).textContent = "Training curves unavailable.";
  }
}

function renderNotebookExperiments(metrics) {
  const summary = $("#notebook-experiment-summary");
  summary.replaceChildren(...[
    ["TEMPORAL VALIDATION ACCURACY", formatPercent(metrics.models?.temporal?.results?.validation?.accuracy)],
    ["TEMPORAL TEST ACCURACY", formatPercent(metrics.models?.temporal?.results?.test?.accuracy)],
    ["REGULARIZED VALIDATION ACCURACY", formatPercent(metrics.models?.regularized?.results?.validation?.accuracy)],
    ["REGULARIZED TEST ACCURACY", formatPercent(metrics.models?.regularized?.results?.test?.accuracy)],
  ].map(([label, value]) => { const card = document.createElement("div"); card.className = "metric-card"; const title = document.createElement("span"); title.textContent = label; const number = document.createElement("strong"); number.textContent = value; card.append(title, number); return card; }));
  renderExperimentCurve("temporal", "notebook-temporal-curve");
  renderExperimentCurve("regularized", "notebook-regularized-curve");
}

function renderNotebookPage(metrics) {
  renderNotebookDataset(metrics);
  renderNotebookFeatures(metrics);
  renderNotebookSplit(metrics);
  renderModelStage("cnn", "notebook-cnn-summary", "notebook-cnn-architecture", "notebook-cnn-curve");
  renderModelStage("cnn_bilstm", "notebook-cnn-bilstm-summary", "notebook-cnn-bilstm-architecture", "notebook-cnn-bilstm-curve");
  renderNotebookComparison(metrics);
  const modelOptions = Object.keys(metrics.models || {});
  const selectModel = $("#notebook-report-model");
  selectModel.innerHTML = "";
  modelOptions.filter((name) => name !== "ensemble").forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name.replaceAll("_", " ").toUpperCase();
    selectModel.appendChild(option);
  });
  $("#notebook-report-split").addEventListener("change", () => renderClassificationReport(metrics));
  selectModel.addEventListener("change", () => renderClassificationReport(metrics));
  renderClassificationReport(metrics);
  renderNotebookExperiments(metrics);
}

async function initialize() {
  try {
    state.health = await api("/api/health");
    const aboutApi = $("#about-api");
    if (aboutApi) aboutApi.textContent = "Connected";
  } catch (error) {
    const aboutApi = $("#about-api");
    if (aboutApi) aboutApi.textContent = "Unavailable";
    const predictButton = $("#predict-button");
    if (predictButton) predictButton.title = error.message;
  }
  try {
    state.modelInfo = await api("/api/model-info");
  } catch (error) {
    state.modelInfo = {
      ready: false,
      detail: `Could not load model information: ${error.message}`
    };
  }
  try {
    state.metrics = await api("/api/metrics");
  } catch (error) {
    state.metrics = null;
    state.metricsError = `Could not load evaluation metrics: ${error.message}`;
  }
  const labels = supportedLabels();
  const emotionList = $("#emotion-list");
  if (emotionList) {
    emotionList.replaceChildren(...labels.map((label) => {
      const item = document.createElement("span");
      const code = Object.entries(EMOTION_CODES).find(([, emotion]) => emotion === label)?.[0];
      const codeElement = document.createElement("b");
      codeElement.textContent = code || "—";
      item.append(codeElement, document.createTextNode(label));
      return item;
    }));
  }
  const classCount = $("#dashboard-class-count");
  if (classCount) classCount.textContent = labels.length ? String(labels.length) : "Not available";
  renderModelStatus();
  renderPerformance(state.modelInfo);
  try {
    renderDataset(await api("/api/samples"));
  } catch (error) {
    renderDataset({
      configured: null,
      directory_exists: null,
      dataset_available: null,
      valid_audio_count: null,
      valid_audio_files: null,
      actor_count: null,
      emotion_counts: {},
      samples: [],
      error: `Could not load dataset status: ${error.message}`,
      detail: error.message
    });
  }
  await loadDashboardSummary();
  updateModelAvailability();
  renderModelSummary();
  if (state.metrics) {
    renderNotebookPage(state.metrics);
  }
  await loadHistory();
  renderEmotionDistribution();
  renderRecentPredictions();
  renderDashboardOverview();
  const ensembleAccuracy = $("#dashboard-ensemble-accuracy");
  if (ensembleAccuracy && state.metrics?.models?.ensemble?.results?.test) {
    const acc = state.metrics.models.ensemble.results.test.accuracy;
    ensembleAccuracy.textContent = Number.isFinite(acc) ? `${(acc * 100).toFixed(1)}%` : "—";
  } else if (ensembleAccuracy) {
    ensembleAccuracy.textContent = "—";
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".page").forEach((page) => page.classList.remove("active"));
    button.classList.add("active");
    $(`#page-${button.dataset.page}`).classList.add("active");
    $("#breadcrumb-page").textContent = button.textContent.trim();
  });
});

document.querySelectorAll("[data-nav]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.nav;
    const navButton = document.querySelector(`.nav-item[data-page="${target}"]`);
    if (navButton) navButton.click();
  });
});

const predictionModel = $("#prediction-model");
if (predictionModel) predictionModel.addEventListener("change", renderPredictionProbabilities);
const analysisModelSelect = $("#analysis-model-select");
if (analysisModelSelect) {
  analysisModelSelect.addEventListener("change", () => {
    renderPredictionProbabilities();
    if (predictionModel) predictionModel.value = analysisModelSelect.value;
  });
}
const sampleAudioSelect = $("#sample-audio-select");
if (sampleAudioSelect) {
  sampleAudioSelect.addEventListener("change", async (event) => {
    const filename = event.target.value;
    if (!filename) return;

    try {
      const response = await fetch(`/api/sample-audio?filename=${encodeURIComponent(filename)}`);
      if (!response.ok) throw new Error("Sample audio is not available.");
      const blob = await response.blob();
      const file = new File([blob], filename, { type: "audio/wav" });
      setFile(file);
    } catch (error) {
      showError(error.message);
    }
  });
}
const analyzeAnother = $("#analyze-another");
if (analyzeAnother) {
  analyzeAnother.addEventListener("click", () => {
    removeFile();
    const audioInput = $("#audio-input");
    if (audioInput) audioInput.focus();
  });
}
const performanceSplit = $("#performance-split");
if (performanceSplit) performanceSplit.addEventListener("change", () => renderPerformance(state.modelInfo));
const performanceMetric = $("#performance-metric");
if (performanceMetric) performanceMetric.addEventListener("change", () => renderPerformance(state.modelInfo));
const confusionModel = $("#confusion-model");
if (confusionModel) confusionModel.addEventListener("change", () => renderPerformance(state.modelInfo));
const historySearch = $("#history-search");
if (historySearch) historySearch.addEventListener("input", () => renderHistory(state.history));
const historyDetailClose = $("#history-detail-close");
if (historyDetailClose) historyDetailClose.addEventListener("click", () => $("#history-detail").classList.add("hidden"));
const historyDetail = $("#history-detail");
if (historyDetail) {
  historyDetail.addEventListener("click", (event) => {
    if (event.target === historyDetail) historyDetail.classList.add("hidden");
  });
}
const audioInput = $("#audio-input");
if (audioInput) audioInput.addEventListener("change", (event) => setFile(event.target.files[0]));
const removeFileButton = $("#remove-file");
if (removeFileButton) removeFileButton.addEventListener("click", removeFile);
const predictButton = $("#predict-button");
if (predictButton) predictButton.addEventListener("click", predict);
const dropZone = $("#drop-zone");
if (dropZone) {
  dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    setFile(event.dataTransfer.files[0]);
  });
}

initialize();

/**
 * =======================================================================================
 * |                        Advanced Logistics Optimization System JS                      |
 * |                                     Version: 3.2                                    |
 * |-------------------------------------------------------------------------------------|
 * |                                                                                     |
 * | This script provides the complete frontend logic for the logistics optimization     |
 * | dashboard. It has been significantly refactored into a robust, object-oriented      |
 * | structure to manage complex state, handle API communications, and dynamically       |
 * | update the user interface.                                                          |
 * |                                                                                     |
 * | Key Architectural Components:                                                       |
 * |  - StateManager: A dedicated class to act as a single source of truth for all       |
 * |                  application data.                                                  |
 * |  - APIClient:    A robust client for handling all network communications with the   |
 * |                  Python FastAPI backend.                                           |
 * |  - UIManager:    Manages all DOM manipulations, UI updates, and event listeners.    |
 * |  - App:          The main application class that orchestrates the entire system.    |
 * |                                                                                     |
 * |=====================================================================================|
 * | **BUG FIX for "Cannot set properties of undefined (setting 'textContent')"**:      |
 * | This version resolves the error that occurred when fetching final results. The     |
 * | issue was that the code attempted to write to a DOM element that was not yet       |
 * | guaranteed to be available. The `updateGlobalBest` and `updateReportView`          |
 * | functions have been made more robust by adding checks to ensure DOM elements       |
 * | exist before attempting to modify their `textContent` or other properties. This    |
 * | prevents the application from crashing if the UI state is not as expected.         |
 * =======================================================================================
 */

/**
 * Manages the application's state, acting as a single source of truth.
 * This prevents state inconsistencies and makes data flow predictable.
 */
class StateManager {
    constructor() {
        this.problemData = null; // Stores the generated problem data (locations, demands)
        this.optimizationResults = null; // Stores the final results from the backend
        this.isOptimizing = false; // Flag to prevent concurrent optimization runs
        this.isGeneratingData = false; // Flag to prevent concurrent data generation
        this.taskId = null; // Stores the ID of the current background optimization task
        this.pollInterval = null; // Holds the interval timer for polling task status
        this.logCount = 0; // Tracks the number of logs received to avoid duplicates
    }

    /**
     * Resets the optimization-related state to its initial values.
     * This is crucial before starting a new optimization run to avoid stale data.
     */
    resetOptimizationState() {
        this.optimizationResults = null;
        this.isOptimizing = false;
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        this.taskId = null;
        this.pollInterval = null;
        this.logCount = 0;
    }
}

/**
 * Handles all communication with the backend FastAPI server.
 * It abstracts away the complexities of fetch() and provides clean, async methods
 * with centralized error handling.
 */
class APIClient {
    /**
     * @param {string} baseUrl The base URL of the FastAPI server.
     */
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    /**
     * A generic, private fetch method that includes robust error handling.
     * @param {string} endpoint The API endpoint to call (e.g., '/generate-data').
     * @param {object} options The options object for the fetch call (method, headers, body).
     * @returns {Promise<any>} A promise that resolves with the response data (JSON or raw).
     * @throws {Error} Throws a detailed error if the request fails or the server returns an error.
     */
    async _fetch(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, options);

            if (!response.ok) {
                let errorMessage = `HTTP Error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    if (errorData && errorData.detail) {
                        errorMessage = Array.isArray(errorData.detail)
                            ? errorData.detail.map(err => `Field '${err.loc.join(' -> ')}': ${err.msg}`).join('; ')
                            : errorData.detail;
                    }
                } catch (e) {
                    // Ignore if the error response itself is not valid JSON.
                }
                throw new Error(errorMessage);
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                return response.json();
            }
            return response; // Return the raw response for non-JSON content like file downloads
        } catch (error) {
            console.error(`API Client Error on '${endpoint}':`, error);
            throw error; // Re-throw to be handled by the calling function
        }
    }

    generateData(params) {
        return this._fetch('/generate-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
    }

    runOptimization(payload) {
        return this._fetch('/run-optimization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    }

    checkStatus(taskId) {
        return this._fetch(`/check-status/${taskId}`);
    }

    getResults(taskId) {
        return this._fetch(`/get-results/${taskId}`);
    }

    loadConfig(formData) {
        return this._fetch('/load-config', { method: 'POST', body: formData });
    }

    saveConfig(params) {
        return this._fetch('/save-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
    }
}

/**
 * Manages all DOM interactions, UI updates, and event handling.
 * This class isolates UI logic from the main application flow.
 */
class UIManager {
    constructor() {
        this.dom = this._gatherDOMElements();
        this.errorModal = new bootstrap.Modal(this.dom.errorModal);
    }

    /**
     * Gathers all necessary DOM elements into a single, easily accessible object.
     * @private
     */
    _gatherDOMElements() {
        const ids = [
            'parametersAccordion', 'loadConfigBtn', 'saveConfigBtn', 'loadConfigFile',
            'generateDataBtn', 'runOptimizationBtn', 'statusText', 'progressBar',
            'progressBarInner', 'cost_weight', 'time_weight', 'cost_weight_label',
            'time_weight_label', 'resultsTabContent', 'welcome-panel',
            'errorModal', 'errorModalBody',
            'global-best-summary', 'best-algo-name', 'best-algo-cost'
        ];
        const elements = {};
        ids.forEach(id => elements[id] = document.getElementById(id));
        elements.algoCheckboxes = document.querySelectorAll('input[id^="algo_"]');
        return elements;
    }

    /**
     * Initializes all primary event listeners for the application.
     * @param {App} app The main application instance to handle the events.
     */
    initializeEventListeners(app) {
        this.dom.generateDataBtn.addEventListener('click', () => app.handleGenerateData());
        this.dom.runOptimizationBtn.addEventListener('click', () => app.handleRunOptimization());
        this.dom.loadConfigBtn.addEventListener('click', () => this.dom.loadConfigFile.click());
        this.dom.loadConfigFile.addEventListener('change', (e) => app.handleLoadConfig(e));

        // Use event delegation on the body for dynamically created elements
        document.body.addEventListener('click', (e) => {
            const target = e.target.closest('button');
            if (!target) return;

            switch (target.id) {
                case 'openMapBtn':
                    const selectedMapUrl = this.dom.mapResultSelect?.value;
                    if (selectedMapUrl && selectedMapUrl !== 'none') window.open(selectedMapUrl, '_blank');
                    break;
                case 'saveIterationPlotBtn':
                    this.downloadContent('#iterationCurveContainer img', 'iteration_curves.png');
                    break;
                case 'saveComparisonPlotBtn':
                    this.downloadContent('#comparisonChartContainer img', 'comparison_chart.png');
                    break;
                case 'saveReportBtn':
                    this.downloadContent('#reportDisplayArea', 'report.txt', 'text/plain');
                    break;
            }
        });

        document.body.addEventListener('change', (e) => {
            if (e.target.id === 'mapResultSelect') {
                this.updateMapFrame(e.target.value);
            } else if (e.target.id === 'reportResultSelect') {
                this.updateReportView(app.state.optimizationResults);
            }
        });

        // Event listeners for range sliders to update their labels in real-time
        this.dom.cost_weight.addEventListener('input', (e) => this.dom.cost_weight_label.textContent = parseFloat(e.target.value).toFixed(2));
        this.dom.time_weight.addEventListener('input', (e) => this.dom.time_weight_label.textContent = parseFloat(e.target.value).toFixed(2));

        console.log("UI Manager: All event listeners have been successfully initialized.");
    }

    /**
     * Dynamically creates the results panel, replacing the welcome message.
     * This is done only once when the first results are ready to be displayed.
     */
    createResultsPanels() {
        // Now that the panels are static in index.html, this function's only job
        // is to remove the welcome panel and re-gather the (now visible) elements.
        const welcomePanel = document.getElementById('welcome-panel');
        if (welcomePanel) {
            welcomePanel.remove();
        }
        // This is still important to get references to the elements that are now part of the DOM.
        this._reGatherDynamicElements();
    }

    /**
     * Gathers references to dynamically created DOM elements after an update.
     * @private
     */
    _reGatherDynamicElements() {
        const ids = ['mapResultSelect', 'openMapBtn', 'mapFrame', 'mapPlaceholder', 'iterationCurveContainer', 'saveIterationPlotBtn', 'comparisonChartContainer', 'saveComparisonPlotBtn', 'reportResultSelect', 'saveReportBtn', 'reportDisplayArea', 'logDisplayArea'];
        ids.forEach(id => this.dom[id] = document.getElementById(id));
    }

    /**
     * Updates the status bar with a message and optional progress.
     * @param {string} text The message to display.
     * @param {number|null} progress The progress percentage (0-100).
     * @param {'info'|'success'|'error'|'warning'} type The type of message for styling.
     */
    updateStatus(text, progress = null, type = 'info') {
        const typeToClass = { info: 'text-muted', success: 'text-success', error: 'text-danger', warning: 'text-warning' };
        this.dom.statusText.textContent = text;
        this.dom.statusText.className = `small ${typeToClass[type] || 'text-muted'}`;
        this.dom.progressBar.style.display = progress !== null ? 'flex' : 'none';
        if(progress !== null) this.dom.progressBarInner.style.width = `${progress}%`;
    }

    /**
     * Disables or enables all user controls during processing.
     * @param {boolean} disabled True to disable controls, false to enable.
     */
    setControlsDisabled(disabled) {
        const elementsToToggle = [this.dom.generateDataBtn, this.dom.runOptimizationBtn, this.dom.loadConfigBtn, this.dom.saveConfigBtn, ...this.dom.algoCheckboxes];
        elementsToToggle.forEach(el => el && (el.disabled = disabled));
        this.dom.parametersAccordion.querySelectorAll('input, select').forEach(input => {
            if (input.id !== 'loadConfigFile') input.disabled = disabled;
        });
        this.toggleSpinner(this.dom.generateDataBtn, app.state.isGeneratingData && disabled);
        this.toggleSpinner(this.dom.runOptimizationBtn, app.state.isOptimizing && disabled);
    }

    /**
     * Shows or hides the spinner icon inside a button.
     * @param {HTMLElement} button The button element.
     * @param {boolean} show True to show the spinner, false to hide.
     */
    toggleSpinner(button, show) {
        if (!button) return;
        const spinner = button.querySelector('.spinner-border');
        if (show) {
            spinner?.classList.remove('d-none');
            button.classList.add('disabled');
        } else {
            spinner?.classList.add('d-none');
            button.classList.remove('disabled');
        }
    }

    showError(message) {
        this.dom.errorModalBody.textContent = message;
        this.errorModal.show();
    }

    log(message, type = 'info') {
        if (!this.dom.logDisplayArea) return;
        if (this.dom.logDisplayArea.classList.contains('text-muted')) {
            this.dom.logDisplayArea.innerHTML = '';
            this.dom.logDisplayArea.classList.remove('text-muted');
        }
        const typeClassMap = { info: 'text-light', success: 'text-success', warning: 'text-warning', error: 'text-danger fw-bold' };
        this.dom.logDisplayArea.innerHTML += `<div class="log-entry ${typeClassMap[type] || 'text-light'}"><span class="text-secondary me-2">[${new Date().toLocaleTimeString()}]</span><span>${message}</span></div>`;
        this.dom.logDisplayArea.scrollTop = this.dom.logDisplayArea.scrollHeight;
    }

    clearLog() {
        if (this.dom.logDisplayArea) {
            this.dom.logDisplayArea.innerHTML = 'Live run log will appear here...';
            this.dom.logDisplayArea.classList.add('text-muted');
        }
    }

    /**
     * Collects all parameters from the UI forms, with type-safe parsing.
     * @returns {object} A structured object containing all parameters.
     */
    collectParameters() {
        const getVal = (id, type = 'string') => {
            const el = document.getElementById(id);
            if (!el) return null;
            if (type === 'bool') return el.checked;
            const value = el.value;
            return type === 'float' ? parseFloat(value) : (type === 'int' ? parseInt(value, 10) : value);
        };
        // This structure must exactly match the Pydantic models in the backend
        return {
            data_generation: {
                num_logistics_centers: getVal('num_logistics_centers', 'int'),
                num_sales_outlets: getVal('num_sales_outlets', 'int'),
                num_customers: getVal('num_customers', 'int'),
                use_solomon_like_distribution: getVal('use_solomon_like_distribution', 'bool'),
                center_latitude: getVal('center_latitude', 'float'),
                center_longitude: getVal('center_longitude', 'float'),
                radius_km: getVal('radius_km', 'float'),
                min_demand: getVal('min_demand', 'float'),
                max_demand: getVal('max_demand', 'float')
            },
            vehicle: {
                max_payload_kg: getVal('vehicle_payload', 'float'),
                cost_per_km: getVal('vehicle_cost', 'float'),
                speed_kmh: getVal('vehicle_speed', 'float')
            },
            drone: {
                max_payload_kg: getVal('drone_payload', 'float'),
                cost_per_km: getVal('drone_cost', 'float'),
                speed_kmh: getVal('drone_speed', 'float'),
                max_flight_distance_km: getVal('drone_range', 'float')
            },
            objective: {
                cost_weight: getVal('cost_weight', 'float'),
                time_weight: getVal('time_weight', 'float'),
                unmet_demand_penalty: getVal('unmet_demand_penalty', 'float')
            },
            algorithm_params: {
                genetic_algorithm: { population_size: getVal('ga_pop_size', 'int'), num_generations: getVal('ga_gens', 'int'), mutation_rate: getVal('ga_mut_rate', 'float'), crossover_rate: getVal('ga_cross_rate', 'float'), elite_count: getVal('ga_elitism', 'int'), tournament_size: getVal('ga_tourn_size', 'int') },
                simulated_annealing: { initial_temperature: getVal('sa_init_temp', 'float'), cooling_rate: getVal('sa_cool_rate', 'float'), max_iterations: getVal('sa_iters', 'int'), min_temperature: getVal('sa_min_temp', 'float') },
                pso_optimizer: { num_particles: getVal('pso_swarm_size', 'int'), max_iterations: getVal('pso_max_iters', 'int'), inertia_weight: getVal('pso_inertia', 'float'), cognitive_weight: getVal('pso_cognitive', 'float'), social_weight: getVal('pso_social', 'float') },
                greedy_heuristic: {}
            },
            selected_algorithms: Array.from(this.dom.algoCheckboxes).filter(cb => cb.checked).map(cb => cb.value)
        };
    }

    /**
     * Applies a configuration object (from a loaded .ini file) to the UI form fields.
     * @param {object} configData The configuration data object.
     */
    applyConfigToUI(configData) {
        const setVal = (id, value) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.type === 'checkbox' ? el.checked = (String(value).toLowerCase() === 'true') : el.value = value;
            if (el.type === 'range') el.dispatchEvent(new Event('input')); // Update slider labels
        };

        const mapConfigToUI = (sectionKey, mapping) => {
            const sectionData = configData[sectionKey];
            if (sectionData) {
                Object.entries(mapping).forEach(([configKey, elementId]) => {
                    if (sectionData[configKey] !== undefined) {
                        setVal(elementId, sectionData[configKey]);
                    }
                });
            }
        };

        mapConfigToUI('data_generation', { num_logistics_centers: 'num_logistics_centers', num_sales_outlets: 'num_sales_outlets', num_customers: 'num_customers', use_solomon_like_distribution: 'use_solomon_like_distribution', center_latitude: 'center_latitude', center_longitude: 'center_longitude', radius_km: 'radius_km', min_demand: 'min_demand', max_demand: 'max_demand' });
        mapConfigToUI('vehicle', { max_payload_kg: 'vehicle_payload', cost_per_km: 'vehicle_cost', speed_kmh: 'vehicle_speed' });
        mapConfigToUI('drone', { max_payload_kg: 'drone_payload', cost_per_km: 'drone_cost', speed_kmh: 'drone_speed', max_flight_distance_km: 'drone_range' });
        mapConfigToUI('objective', { cost_weight: 'cost_weight', time_weight: 'time_weight', unmet_demand_penalty: 'unmet_demand_penalty' });

        if (configData.algorithm_params) {
            const algoMappings = {
                genetic_algorithm: { population_size: 'ga_pop_size', num_generations: 'ga_gens', mutation_rate: 'ga_mut_rate', crossover_rate: 'ga_cross_rate', elite_count: 'ga_elitism', tournament_size: 'ga_tourn_size' },
                simulated_annealing: { initial_temperature: 'sa_init_temp', cooling_rate: 'sa_cool_rate', max_iterations: 'sa_iters', min_temperature: 'sa_min_temp' },
                pso_optimizer: { num_particles: 'pso_swarm_size', max_iterations: 'pso_max_iters', inertia_weight: 'pso_inertia', cognitive_weight: 'pso_cognitive', social_weight: 'pso_social' }
            };

            // (e.g., genetic_algorithm, pso_optimizer)
            Object.entries(configData.algorithm_params).forEach(([algoKey, paramsObject]) => {
                const mapping = algoMappings[algoKey];
                //
                if (mapping && typeof paramsObject === 'object') {
                    //  (e.g., population_size, cognitive_weight)
                    Object.entries(paramsObject).forEach(([paramKey, paramValue]) => {
                        const elementId = mapping[paramKey]; //
                        if (elementId) {
                            setVal(elementId, paramValue); //
                        }
                    });
                }
            });
        }
        this.log('Configuration successfully loaded and applied to UI.', 'success');
    }

    /**
     * Populates all result views (maps, plots, reports) after an optimization run.
     * @param {object} results The complete results object from the backend.
     * @param {APIClient} apiClient The API client instance for building URLs.
     */
    populateResults(results, apiClient) {
        if (!results?.results_by_algorithm) return;

        const successfulRuns = Object.entries(results.results_by_algorithm)
            .filter(([_, r]) => r.status === 'Success' && r.result_data);

        if (successfulRuns.length === 0) {
            this.log("Optimization finished, but no algorithms succeeded.", 'warning');
            return;
        }

        this._populateSelect(this.dom.mapResultSelect, successfulRuns, 'map', 'Select a map to view...', results.run_timestamp, apiClient);
        this._populateSelect(this.dom.reportResultSelect, successfulRuns, 'report', 'Select a report to view...', results.run_timestamp, apiClient);

        const bestKey = results.fully_served_best_key || results.best_algorithm_key;
        if (bestKey) {
            const bestResult = results.results_by_algorithm[bestKey];
            if (bestResult?.map_path) {
                const bestMapUrl = `${apiClient.baseUrl}/outputs/${results.run_timestamp}/maps/${bestResult.map_path.split(/[\\/]/).pop()}`;
                if (this.dom.mapResultSelect) {
                    this.dom.mapResultSelect.value = bestMapUrl;
                }
            }
            if (this.dom.reportResultSelect) {
                this.dom.reportResultSelect.value = bestKey;
            }
        }

        this.updateMapFrame(this.dom.mapResultSelect?.value);
        this.updateReportView(results);
        this.updatePlotViews(results, apiClient);
        this.updateGlobalBest(results);
    }

    /**
     * Updates the global best solution summary display.
     * @param {object} results The complete results object.
     * @private
     */
    updateGlobalBest(results) {
        const bestKey = results?.best_algorithm_key;
        if (bestKey && this.dom.globalBestSummary) {
            const bestResult = results.results_by_algorithm[bestKey];
            if (bestResult?.result_data) {
                if (this.dom.bestAlgoName) {
                    this.dom.bestAlgoName.textContent = bestResult.algorithm_name || bestKey;
                }
                if (this.dom.bestAlgoCost) {
                    this.dom.bestAlgoCost.textContent = `Cost: ${bestResult.result_data.weighted_cost.toFixed(2)}`;
                }
                this.dom.globalBestSummary.classList.remove('d-none');
            }
        } else if (this.dom.globalBestSummary) {
            this.dom.globalBestSummary.classList.add('d-none');
        }
    }

    /**
     * Populates a <select> dropdown with results.
     * @private
     */
    _populateSelect(selectElement, runs, type, defaultOptionText, timestamp, apiClient) {
        if (!selectElement) return; // Add guard
        selectElement.innerHTML = `<option value="none">${defaultOptionText}</option>`;
        runs.forEach(([key, result]) => {
            const displayName = result.algorithm_name || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let value;
            if (type === 'map' && result.map_path) {
                const filename = result.map_path.split(/[\\/]/).pop();
                value = `${apiClient.baseUrl}/outputs/${timestamp}/maps/${filename}`;
            } else if (type === 'report') {
                value = key; // Use the algorithm key as the value for reports
            }
            if (value) selectElement.add(new Option(displayName, value));
        });
        selectElement.disabled = false;
    }

    /**
     * Updates the map iframe with the selected URL.
     * @param {string} url The URL of the map to display.
     */
    updateMapFrame(url) {
        const { mapPlaceholder, mapFrame, openMapBtn } = this.dom;
        if (!mapFrame || !mapPlaceholder || !openMapBtn) return; // Add guard

        if (url && url !== 'none') {
            mapPlaceholder.style.display = 'flex';
            const placeholderText = mapPlaceholder.querySelector('.placeholder-text');
            if (placeholderText) placeholderText.textContent = 'Loading map...';

            mapFrame.style.opacity = 0;
            mapFrame.onload = () => { mapPlaceholder.style.display = 'none'; mapFrame.style.opacity = 1; };
            mapFrame.src = url;
            openMapBtn.disabled = false;
        } else {
            mapPlaceholder.style.display = 'flex';
            const placeholderText = mapPlaceholder.querySelector('.placeholder-text');
            if (placeholderText) placeholderText.textContent = 'Select a result to view map.';

            mapFrame.src = 'about:blank';
            openMapBtn.disabled = true;
        }
    }

    /**
     * Fetches and displays the content of the selected report.
     * @param {object} results The complete results object.
     */
    async updateReportView(results) {
        const { reportResultSelect, reportDisplayArea, saveReportBtn } = this.dom;
        if (!reportResultSelect || !reportDisplayArea || !saveReportBtn) return; // Add guard

        const selectedKey = reportResultSelect.value;
        if (selectedKey && selectedKey !== 'none' && results) {
            const reportPath = results.results_by_algorithm[selectedKey]?.report_path;
            if (reportPath) {
                const reportUrl = `${app.api.baseUrl}/outputs/${results.run_timestamp}/reports/${reportPath.split(/[\\/]/).pop()}`;
                try {
                    const response = await fetch(reportUrl);
                    if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);
                    reportDisplayArea.textContent = await response.text();
                    reportDisplayArea.classList.remove('text-muted');
                    saveReportBtn.disabled = false;
                } catch (error) {
                    reportDisplayArea.textContent = `Error loading report: ${error.message}`;
                    saveReportBtn.disabled = true;
                }
            }
        } else {
            reportDisplayArea.textContent = 'Select a result to view the detailed report.';
            reportDisplayArea.classList.add('text-muted');
            saveReportBtn.disabled = true;
        }
    }

    /**
     * Loads the iteration and comparison plot images into their containers.
     * @param {object} results The complete results object.
     * @param {APIClient} apiClient The API client instance.
     */
    updatePlotViews(results, apiClient) {
        const updatePlot = (containerId, buttonId, chartName) => {
            const container = document.getElementById(containerId);
            const button = document.getElementById(buttonId);
            if (!container || !button) return;

            const imageUrl = `${apiClient.baseUrl}/outputs/${results.run_timestamp}/charts/${chartName}?t=${new Date().getTime()}`;
            container.innerHTML = `<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>`;
            const img = new Image();
            img.className = 'img-fluid results-content';
            img.onload = () => { container.innerHTML = ''; container.appendChild(img); button.disabled = false; };
            img.onerror = () => { container.innerHTML = `<span class="text-muted">Could not load chart: ${chartName}</span>`; button.disabled = true; };
            img.src = imageUrl;
        };
        updatePlot('iterationCurveContainer', 'saveIterationPlotBtn', 'iteration_curves.png');
        updatePlot('comparisonChartContainer', 'saveComparisonPlotBtn', 'comparison_chart.png');
    }

    /**
     * Triggers a download for a given piece of content (image or text).
     * @param {string} selector CSS selector for the content element.
     * @param {string} filename The desired filename for the download.
     * @param {string} mimeType The MIME type for text-based downloads.
     */
    downloadContent(selector, filename, mimeType = 'image/png') {
        const element = document.querySelector(selector);
        if (!element) { this.showError("Content is not available for download."); return; }

        if (element.tagName === 'IMG') {
            fetch(element.src)
                .then(res => res.blob())
                .then(blob => this._triggerDownload(blob, filename))
                .catch(err => this.showError(`Download failed: ${err.message}`));
        } else {
            const blob = new Blob([element.textContent], { type: mimeType });
            this._triggerDownload(blob, filename);
        }
    }

    /**
     * Creates a temporary link and clicks it to trigger a file download.
     * @param {Blob} blob The data blob to download.
     * @param {string} filename The name of the file.
     * @private
     */
    _triggerDownload(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
}


/**
 * The main application class. It orchestrates the StateManager, APIClient,
 * and UIManager to create a functional application.
 */
class App {
    constructor() {
        this.state = new StateManager();
        this.api = new APIClient('http://127.0.0.1:8000');
        this.ui = new UIManager();
        this.ui.initializeEventListeners(this);
        this.ui.updateStatus("Ready. Please generate data or load a configuration.", null, 'info');
    }

    async handleGenerateData() {
        if (this.state.isGeneratingData || this.state.isOptimizing) return;

        this.state.isGeneratingData = true;
        this.ui.setControlsDisabled(true);
        this.ui.updateStatus("Generating new data scenario...", null, 'info');
        this.ui.log("Data generation process started...", 'info');
        this.ui.createResultsPanels();

        const params = this.ui.collectParameters().data_generation;

        try {
            const response = await this.api.generateData(params);

            this.state.problemData = response.problem_data;

            const message = `Data generated for ${this.state.problemData.demands.length} customers. Ready for optimization.`;
            this.ui.updateStatus(message, null, 'success');
            this.ui.log(message, 'success');

            if (response.initial_map_path) {
                const mapUrl = `${this.api.baseUrl}${response.initial_map_path}`;
                this.ui.updateMapFrame(mapUrl);
                this.ui.log("Initial layout map has been generated and displayed.", 'info');

                const { mapResultSelect } = this.ui.dom;
                if (mapResultSelect) {
                    mapResultSelect.innerHTML = `<option value="${mapUrl}">Initial Layout</option>`;
                    mapResultSelect.disabled = false;
                }
            }

        } catch (error) {
            this.ui.showError(`Data generation failed: ${error.message}`);
            this.ui.updateStatus("Data generation failed. Please check parameters.", null, 'error');
            this.ui.log(`Data generation failed: ${error.message}`, 'error');
        } finally {
            this.state.isGeneratingData = false;
            this.ui.setControlsDisabled(this.state.isOptimizing);
        }
    }

    async handleRunOptimization() {
        if (!this.state.problemData) {
            this.ui.showError("Cannot run optimization. Please generate data first.");
            return;
        }
        if (this.state.isOptimizing) return;

        const params = this.ui.collectParameters();
        if (params.selected_algorithms.length === 0) {
            this.ui.showError("Please select at least one algorithm to run.");
            return;
        }

        this.state.resetOptimizationState();
        this.state.isOptimizing = true;
        this.ui.createResultsPanels();
        this.ui.setControlsDisabled(true);
        this.ui.clearLog();
        this.ui.updateStatus("Starting optimization task...", 0, 'info');
        this.ui.log(`Starting optimization with: ${params.selected_algorithms.join(', ')}.`, 'info');

        const payload = { ...params, problem_data: this.state.problemData };

        try {
            const response = await this.api.runOptimization(payload);
            this.state.taskId = response.task_id;
            this.ui.updateStatus("Task submitted. Polling for status...", 5, 'info');
            this.ui.log(`Backend task started with ID: ${this.state.taskId}`, 'success');
            this.state.pollInterval = setInterval(() => this.pollTaskStatus(), 2000); // Poll every 2 seconds
        } catch (error) {
            this.ui.showError(`Failed to start optimization task: ${error.message}`);
            this.ui.updateStatus("Optimization failed to start.", null, 'error');
            this.ui.log(`Failed to start optimization task: ${error.message}`, 'error');
            this.state.isOptimizing = false;
            this.ui.setControlsDisabled(false);
        }
    }

    async pollTaskStatus() {
        if (!this.state.taskId) return;
        try {
            const data = await this.api.checkStatus(this.state.taskId);
            this.ui.updateStatus(data.message || 'Processing...', data.progress, 'info');

            if (data.logs && data.logs.length > this.state.logCount) {
                const newLogs = data.logs.slice(this.state.logCount);
                newLogs.forEach(log => this.ui.log(log.substring(log.indexOf(']') + 2), 'info'));
                this.state.logCount = data.logs.length;
            }

            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(this.state.pollInterval);
                this.state.pollInterval = null;
                if (data.status === 'completed') {
                    this.ui.log("Backend task completed. Fetching final results...", 'success');
                    await this.fetchResults();
                } else {
                    throw new Error(data.error || 'Optimization task failed on the backend.');
                }
            }
        } catch (error) {
            clearInterval(this.state.pollInterval);
            this.state.pollInterval = null;
            this.ui.showError(`Task execution error: ${error.message}`);
            this.ui.updateStatus("Error during optimization.", null, 'error');
            this.ui.log(`Task Error: ${error.message}`, 'error');
            this.state.isOptimizing = false;
            this.ui.setControlsDisabled(false);
        }
    }

    async fetchResults() {
        this.ui.updateStatus("Task finished. Fetching final results...", 95, 'info');
        try {
            const results = await this.api.getResults(this.state.taskId);
            this.state.optimizationResults = results;
            this.ui.populateResults(results, this.api);
            this.ui.updateStatus("Results loaded successfully.", 100, 'success');
            this.ui.log("All results processed and loaded into the UI.", 'success');
        } catch (error) {
            this.ui.showError(`Failed to fetch final results: ${error.message}`);
            this.ui.updateStatus("Failed to load results.", null, 'error');
            this.ui.log(`Failed to load final results: ${error.message}`, 'error');
        } finally {
            this.state.isOptimizing = false;
            this.ui.setControlsDisabled(false);
        }
    }

    async handleLoadConfig(event) {
        const file = event.target.files[0];
        if (!file || !file.name.endsWith('.ini')) {
            this.ui.showError("Invalid file. Please upload a .ini file.");
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        this.ui.updateStatus('Loading configuration...', null, 'info');
        try {
            const configData = await this.api.loadConfig(formData);
            this.ui.applyConfigToUI(configData);
            this.ui.updateStatus('Configuration loaded successfully.', null, 'success');
        } catch (error) {
            this.ui.showError(`Error loading config: ${error.message}`);
            this.ui.updateStatus('Failed to load configuration.', null, 'error');
        }
        event.target.value = ''; // Reset file input
    }

    async handleSaveConfig() {
        const params = this.ui.collectParameters();
        const payload = {
            data_generation: params.data_generation,
            vehicle: params.vehicle,
            drone: params.drone,
            objective: params.objective,
            algorithm_params: params.algorithm_params,
        };
        try {
            const response = await this.api.saveConfig(payload);
            const blob = await response.blob();
            this.ui._triggerDownload(blob, `config_${new Date().toISOString().slice(0,10)}.ini`);
            this.ui.log('Configuration file has been saved.', 'success');
        } catch (error) {
            this.ui.showError(`Error saving config: ${error.message}`);
            this.ui.log(`Error saving config: ${error.message}`, 'error');
        }
    }
}

// Global instance of the App, initialized after the DOM is fully loaded.
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new App();
});
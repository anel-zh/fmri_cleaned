%% Script Remarks
% Visualization and summary of fMRI prediction model results
%
% Goals:
%     - Load saved model results
%     - Visualize:
%           * validation-based PC selection
%           * observed vs predicted outcomes
%           * grouped task / task_rest predictions
%           * model weights
%     - Create summary tables across all comparison models
%
% Notes:
%     - Visualization functions operate on the results structure produced by
%       ModelTrainer
%     - ratings_all is preserved for plotting convenience

%% Content
% 1. Initialize configuration.
% 2. Load results structure.
% 3. Create example plots.
% 4. Create and save model summary tables.

clc
clear

%% 1. Initialize configuration
config = PipelineConfig();

config.Participants = { ...
    'sub-001', ...
    'sub-002', ...
    'sub-003', ...
    'sub-004', ...
    'sub-005'};

config.ModelSplit.train_ids = {'sub-001', 'sub-002', 'sub-003'};
config.ModelSplit.valid_ids = {'sub-004'};
config.ModelSplit.test_ids  = {'sub-005'};

config.validate_all();

if config.Verbose
    config.print_summary();
end

%% 2. Load results structure
results_path = fullfile(config.ResultsDir, 'results_structure_public_example.mat');

if ~exist(results_path, 'file')
    error('Results structure file not found: %s', results_path);
end

S = load(results_path);
results_structure = S.results_structure;

%% 3. Initialize visualizer
visualizer = ResultVisualizer(config);

% -------------------------------------------------------------------------
% Example model to inspect
% Change as needed:
%   'dcc_task'
%   'dcc_task_rest'
%   'hrf_voxel_task'
%   'hrf_voxel_task_rest'
%   'combined_task'
%   'combined_task_rest'
% -------------------------------------------------------------------------
model_name = 'combined_task_rest';

%% 4. Example plots
% Validation PC selection
visualizer.plot_pc_selection(results_structure, model_name, ...
    'Metric', 'mean_r', ...
    'ShowBestPC', true);

% Observed vs predicted on held-out test set
visualizer.plot_observed_vs_predicted(results_structure, model_name, 'testing');

% Grouped prediction plot
visualizer.plot_grouped_predictions(results_structure, model_name, 'testing');

% Final selected weights
visualizer.plot_model_weights(results_structure, model_name, ...
    'Source', 'testing', ...
    'MaxPoints', 5000);

%% 5. Summary tables across all models
validation_summary = visualizer.summarize_all_models(results_structure, 'validation');
testing_summary = visualizer.summarize_all_models(results_structure, 'testing');

save(fullfile(config.ResultsDir, 'validation_summary_public_example.mat'), 'validation_summary', '-v7.3');
save(fullfile(config.ResultsDir, 'testing_summary_public_example.mat'), 'testing_summary', '-v7.3');

try
    writetable(validation_summary, fullfile(config.ResultsDir, 'validation_summary_public_example.csv'));
    writetable(testing_summary, fullfile(config.ResultsDir, 'testing_summary_public_example.csv'));
catch
    warning('Could not save one or more summary CSV files.');
end

%% Script Remarks
% Example pipeline script for model development, validation, and testing
%
% Goal:
%     - Train all final comparison models internally through ModelTrainer
%     - Create structure with the following results:
%           * training
%           * validation
%           * testing
%           * best PC selected from validation
%           * separate storage of weights, stats, and outputs
%
% Final comparison models:
%     1. dcc_task
%     2. dcc_task_rest
%     3. hrf_voxel_task
%     4. hrf_voxel_task_rest
%     5. combined_task
%     6. combined_task_rest
%
% Notes:
%     - ModelTrainer loops over all model conditions internally
%     - X is stored as features x samples
%     - one fold = one scan unit

%% Content
% 1. Initialize configuration.
% 2. Load prepared data.
% 3. Run model training / validation / testing.
% 4. Save results structure.

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

%% 2. Load prepared data
prepared_data_path = fullfile(config.ResultsDir, 'prepared_data_public_example.mat');

if ~exist(prepared_data_path, 'file')
    error('Prepared data file not found: %s', prepared_data_path);
end

S = load(prepared_data_path);
prepared_data = S.prepared_data;

%% 3. Run model training
trainer = ModelTrainer(config);

results_structure = trainer.train_all_models( ...
    prepared_data, ...
    'SaveFolderName', 'public_example_models', ...
    'PerformanceMetric', 'mean_r', ...
    'SaveResults', true);

%% 4. Save results structure
save(fullfile(config.ResultsDir, 'results_structure_public_example.mat'), ...
    'results_structure', '-v7.3');

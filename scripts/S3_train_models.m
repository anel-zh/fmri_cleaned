%% Script Remarks
% Model training, validation, and testing for fMRI prediction models (6 final models) 
%
% Goals:
%     - Train predictive models using cross-validation
%     - Select optimal model complexity using validation data
%     - Evaluate final model performance on the test dataset
%     - Save model weights, predictions, and evaluation metrics
%
% Notes:
%     - Models are trained for multiple feature configurations
%     - Results are saved in a structured format for later analysis
%     - Participants and split options are defined for a small demonstration dataset
%     - In this public example:
%         - one participant / scan unit = one fold unit in modeling
%
%% Content
% 1. Initialize pipeline configuration.
% 2. Load prepared datasets.
% 3. Run model development.
% 4. Save model results.

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

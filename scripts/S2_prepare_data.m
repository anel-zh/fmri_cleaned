%% Script Remarks
% Preparation of model-ready datasets from extracted fMRI features
%
% Goals:
%     - Load extracted feature files and behavioral targets
%     - Split data into train, validation, and test sets
%     - Bin connectivity features
%     - Apply variance normalization after binning
%     - Construct final model comparison datasets
%
% Notes:
%     - Final comparison models include:
%           dcc_task
%           dcc_task_rest
%           hrf_voxel_task
%           hrf_voxel_task_rest
%           combined_task
%           combined_task_rest
%     - Output datasets are used by ModelTrainer
%     - Participants and split options are defined for a small demonstration dataset
%     - In this public example:
%         - one participant / scan unit = one fold unit later in modeling
%
%% Content
% 1. Initialize pipeline configuration.
% 2. Load extracted feature files.
% 3. Assemble model-ready datasets.
% 4. Save prepared data structure.


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

%% 2. Initialize assembler
assembler = DataAssembler(config);

%% 3. Prepare model-ready datasets
prepared_data = assembler.assemble_data( ...
    'FolderSuffix', 'public_example', ...
    'NormType', 'wholebrain', ...
    'SaveIntermediate', true);

%% 4. Save prepared data
save(fullfile(config.ResultsDir, 'prepared_data_public_example.mat'), ...
    'prepared_data', '-v7.3');

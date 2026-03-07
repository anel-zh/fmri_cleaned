%% Script Remarks
% Example pipeline script for preparing extracted fMRI features into
% model-ready datasets
%
% Goal:
%     - Load extracted task and rest feature outputs from disk
%     - Split data into train / valid / test by scan unit
%     - Bin connectivity after split
%     - Apply variance normalization after binning for both:
%           * dcc
%           * hrf_voxel
%     - Assemble final comparison datasets for:
%           1. dcc_task
%           2. dcc_task_rest
%           3. hrf_voxel_task
%           4. hrf_voxel_task_rest
%           5. combined_task
%           6. combined_task_rest
%
% Notes:
%     - Combined models use:
%           dcc + hrf_voxel
%     - HRF ROI is not used as a final modeling branch
%     - X is stored as features x samples
%     - one fold = one scan unit

%% Content
% 1. Initialize configuration.
% 2. Define participants and split.
% 3. Prepare model-ready datasets.
% 4. Save prepared data.

clc
clear

%% 1. Initialize configuration
config = PipelineConfig();

config.Participants = { ... % example set-up of IDs
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

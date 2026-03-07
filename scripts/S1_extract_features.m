%% Script Remarks
% Feature extraction from preprocessed fMRI runs
%
% Goals:
%     - Load task and rest fMRI runs
%     - Extract dynamic connectivity features using DCC
%     - Extract HRF-based activation features
%     - Save extracted feature matrices for modeling
%
% Notes:
%     - Task run corresponds to a capsaicin sustained-pain run
%     - Rest run corresponds to a resting-state run
%     - Extracted features are later assembled by DataAssembler
%     - Participants and split options are defined for a small demonstration dataset
%     - In this public example:
%         - one participant / scan unit = one fold unit later in modeling
%
%% Content
% 1. Initialize pipeline configuration.
% 2. Define participants and dataset split.
% 3. Run feature extraction.
% 4. Save extracted feature files.

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

% Example split definition
config.ModelSplit.train_ids = {'sub-001', 'sub-002', 'sub-003'};
config.ModelSplit.valid_ids = {'sub-004'};
config.ModelSplit.test_ids  = {'sub-005'};

% Validate config
config.validate_all();

if config.Verbose
    config.print_summary();
end

%% 2. Initialize extractor
extractor = FMRIFeatureExtractor(config);

%% 3. Run feature extraction
% Supported extraction methods:
%   - dcc
%   - hrf_roi
%   - hrf_voxel
%
% Notes:
%   - hrf_roi is kept because ROI-level extraction is needed for DCC derivation
%   - final modeling uses:
%         dcc
%         hrf_voxel
%         combined = dcc + hrf_voxel

extractor.extract_features( ...
    'Methods', {'dcc', 'hrf_roi', 'hrf_voxel'}, ...
    'Runs', {'task', 'rest'}, ...
    'ApplyDurationCut', true, ...
    'FolderSuffix', 'public_example');

%% 4. Save configuration summary
save(fullfile(config.ResultsDir, 'config_used_for_feature_extraction.mat'), 'config', '-v7.3');

%% Script Remarks
% Supplementary script for behavioral data processing
% Participants and split options are based on small example sample for demonstration purposes
%
% Goal:
%     - Load raw continuous behavioral ratings
%     - Interpolate ratings to TR
%     - Apply task/rest timing rules
%     - Bin ratings into final model-ready behavioral targets
%     - Save outputs as:
%           task_behavioral_ratings
%           rest_behavioral_ratings
%
% Notes:
%     - Task run corresponds to a capsaicin sustained-pain run
%     - Rest run corresponds to a resting-state run
%     - Saved outputs are later loaded by DataAssembler

%% Content:
% 1. Initialize configuration.
% 2. Define participants and split.
% 3. Prepare behavioral targets.
% 4. Save processed behavioral structure.

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

%% 2. Initialize behavioral processor
behavioral_processor = BehavioralProcessor(config);

%% 3. Prepare behavioral targets
processed_behavioral = behavioral_processor.prepare_all( ...
    'SaveResults', true);

%% 4. Save processed behavioral structure
save(fullfile(config.ResultsDir, 'processed_behavioral_public_example.mat'), ...
    'processed_behavioral', '-v7.3');

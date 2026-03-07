classdef PipelineConfig < handle
    % PIPELINECONFIG centralizes configuration settings for the modular fMRI prediction pipeline
    %
    % Goals:
    %   - Store project paths and core pipeline settings
    %   - Define task and rest run timing parameters
    %   - Specify feature extraction and model comparison settings
    %   - Manage train, validation, and test dataset splits
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   config.Participants = {'sub-001','sub-002'};
    %   config.validate_all();
    %
    % Inputs:
    %   - Optional configuration overrides defined by the user
    %
    % Outputs:
    %   - Configuration object used by all pipeline modules
    %
    % Author: Anel Zhunussova

    properties
        %% Directory Paths
        ProjectRoot
        RawDataDir
        ResultsDir
        FiguresDir
        MetadataDir

        %% Study Details
        StudyName
        ExampleParadigm
        TaskDescription
        RestDescription
        TR = 0.46

        %% Imaging Assets
        AtlasName
        MaskName

        %% Run Discovery
        TaskPattern
        RestPattern
        DenoisingMethod

        %% Participants and Runs
        Participants
        RunLabels

        %% Feature Extraction
        SupportedMethods
        DefaultMethods

        %% Behavioral Targets
        TaskBehavioralField
        RestBehavioralField

        %% Run-Specific Timing / Binning
        RunConfig

        %% Modeling Preferences
        AnalysisMode
        GroupingVariable
        CVScheme
        ModelFeatureSets
        ModelRunConditions
        ModelComparisonNames
        ModelSplit

        %% General Processing Options
        SaveIntermediate = true
        OverwriteOutputs = false
        Verbose = true
    end

    methods
        function obj = PipelineConfig(varargin)
            %% Default folder structure
            obj.ProjectRoot = '.';
            obj.RawDataDir = fullfile(obj.ProjectRoot, 'data', 'preprocessed');
            obj.ResultsDir = fullfile(obj.ProjectRoot, 'results');
            obj.FiguresDir = fullfile(obj.ProjectRoot, 'figures');
            obj.MetadataDir = fullfile(obj.ProjectRoot, 'data', 'metadata');

            %% Default study details
            obj.StudyName = 'Example_fMRI_TimeSeries_Pipeline';
            obj.ExampleParadigm = 'capsaicin_sustained_pain';
            obj.TaskDescription = 'Capsaicin sustained-pain task run';
            obj.RestDescription = 'Resting-state run';

            %% Default imaging assets
            obj.AtlasName = 'Schaefer_265';
            obj.MaskName = 'gray_matter_mask.nii';

            %% Default run discovery
            obj.TaskPattern = '*task*_bold.nii';
            obj.RestPattern = '*rest*_bold.nii';
            obj.DenoisingMethod = 'standard';

            %% Default participants and runs
            obj.Participants = {};
            obj.RunLabels = {'task', 'rest'};

            %% Default feature extraction options
            obj.SupportedMethods = {'dcc', 'hrf_roi', 'hrf_voxel'};
            obj.DefaultMethods = {'dcc', 'hrf_roi', 'hrf_voxel'};

            %% Default behavioral targets
            obj.TaskBehavioralField = 'task_behavioral_ratings';
            obj.RestBehavioralField = 'rest_behavioral_ratings';

            %% Default modeling preferences
            obj.AnalysisMode = 'subjectwise';
            obj.GroupingVariable = 'scan_unit_id';
            obj.CVScheme = 'loso';

            obj.ModelFeatureSets = {'dcc', 'hrf_voxel', 'combined'};
            obj.ModelRunConditions = {'task', 'task_rest'};
            obj.ModelComparisonNames = { ...
                'dcc_task', ...
                'dcc_task_rest', ...
                'hrf_voxel_task', ...
                'hrf_voxel_task_rest', ...
                'combined_task', ...
                'combined_task_rest'};

            obj.ModelSplit = struct();
            obj.ModelSplit.train_ids = {};
            obj.ModelSplit.valid_ids = {};
            obj.ModelSplit.test_ids = {};

            %% Default run timing configuration
            obj.RunConfig = struct();

            obj.RunConfig.task = struct( ...
                'Label', 'task', ...
                'Description', 'Capsaicin sustained-pain task run', ...
                'TotalTR', 1510, ...
                'StartTR', 1, ...
                'MaxTR', 1510, ...
                'WindowSizeTR', 90, ...
                'NumBins', 15, ...
                'RemoveStimulusDelivery', true, ...
                'StimulusDeliveryStartSec', 30, ...
                'StimulusDeliveryEndSec', 70 ...
            );

            obj.RunConfig.rest = struct( ...
                'Label', 'rest', ...
                'Description', 'Resting-state run', ...
                'TotalTR', 1510, ...
                'StartTR', 1, ...
                'MaxTR', 1510, ...
                'WindowSizeTR', 90, ...
                'NumBins', 15 ...
            );

            %% Optional overrides
            if ~isempty(varargin)
                if mod(numel(varargin), 2) ~= 0
                    error('PipelineConfig inputs must be provided as name-value pairs.');
                end

                for i = 1:2:numel(varargin)
                    prop_name = varargin{i};
                    prop_val = varargin{i + 1};

                    if isprop(obj, prop_name)
                        obj.(prop_name) = prop_val;
                    else
                        error('Unknown PipelineConfig property: %s', prop_name);
                    end
                end
            end
        end

        function validate_paths(obj)
            if ~exist(obj.RawDataDir, 'dir')
                error('RawDataDir not found: %s', obj.RawDataDir);
            end
            if ~exist(obj.ResultsDir, 'dir')
                mkdir(obj.ResultsDir);
            end
            if ~exist(obj.FiguresDir, 'dir')
                mkdir(obj.FiguresDir);
            end
            if ~exist(obj.MetadataDir, 'dir')
                mkdir(obj.MetadataDir);
            end
        end

        function validate_core_fields(obj)
            required_text = {'StudyName', 'ExampleParadigm', ...
                             'TaskDescription', 'RestDescription', ...
                             'AtlasName', 'MaskName', ...
                             'TaskPattern', 'RestPattern', ...
                             'DenoisingMethod', ...
                             'TaskBehavioralField', 'RestBehavioralField', ...
                             'AnalysisMode', 'GroupingVariable', 'CVScheme'};

            for i = 1:numel(required_text)
                value = obj.(required_text{i});
                if ~(ischar(value) || isstring(value)) || strlength(string(value)) == 0
                    error('Invalid or empty config field: %s', required_text{i});
                end
            end

            if ~iscell(obj.Participants)
                error('Participants must be a cell array.');
            end

            if isempty(obj.RunLabels) || ~iscell(obj.RunLabels)
                error('RunLabels must be a non-empty cell array.');
            end

            if isempty(obj.SupportedMethods) || ~iscell(obj.SupportedMethods)
                error('SupportedMethods must be a non-empty cell array.');
            end

            if isempty(obj.DefaultMethods) || ~iscell(obj.DefaultMethods)
                error('DefaultMethods must be a non-empty cell array.');
            end

            if ~isscalar(obj.TR) || ~isnumeric(obj.TR) || obj.TR <= 0
                error('TR must be a positive numeric scalar.');
            end

            if ~isstruct(obj.RunConfig) || isempty(fieldnames(obj.RunConfig))
                error('RunConfig must be a non-empty struct.');
            end

            if isempty(obj.ModelFeatureSets) || ~iscell(obj.ModelFeatureSets)
                error('ModelFeatureSets must be a non-empty cell array.');
            end

            if isempty(obj.ModelRunConditions) || ~iscell(obj.ModelRunConditions)
                error('ModelRunConditions must be a non-empty cell array.');
            end

            if isempty(obj.ModelComparisonNames) || ~iscell(obj.ModelComparisonNames)
                error('ModelComparisonNames must be a non-empty cell array.');
            end

            if ~isstruct(obj.ModelSplit)
                error('ModelSplit must be a struct.');
            end
        end

        function validate_run_config(obj)
            for i = 1:numel(obj.RunLabels)
                run_name = obj.RunLabels{i};

                if ~isfield(obj.RunConfig, run_name)
                    error('RunConfig is missing an entry for run label: %s', run_name);
                end

                cfg = obj.RunConfig.(run_name);

                required_fields = {'Label', 'Description', 'TotalTR', ...
                                   'StartTR', 'MaxTR', 'WindowSizeTR', 'NumBins'};

                for j = 1:numel(required_fields)
                    if ~isfield(cfg, required_fields{j})
                        error('RunConfig.%s is missing field: %s', run_name, required_fields{j});
                    end
                end
            end
        end

        function validate_method_config(obj)
            invalid_methods = setdiff(obj.DefaultMethods, obj.SupportedMethods);
            if ~isempty(invalid_methods)
                error('DefaultMethods contains unsupported methods: %s', strjoin(invalid_methods, ', '));
            end
        end

        function validate_model_split(obj)
            required_split_fields = {'train_ids', 'valid_ids', 'test_ids'};

            for i = 1:numel(required_split_fields)
                if ~isfield(obj.ModelSplit, required_split_fields{i})
                    error('ModelSplit is missing field: %s', required_split_fields{i});
                end
                if ~iscell(obj.ModelSplit.(required_split_fields{i}))
                    error('ModelSplit.%s must be a cell array.', required_split_fields{i});
                end
            end
        end

        function validate_all(obj)
            obj.validate_paths();
            obj.validate_core_fields();
            obj.validate_run_config();
            obj.validate_method_config();
            obj.validate_model_split();
        end

        function cfg = get_run_config(obj, run_label)
            run_fields = fieldnames(obj.RunConfig);
            match_idx = strcmpi(run_fields, run_label);

            if ~any(match_idx)
                error('RunConfig entry not found for run label: %s', run_label);
            end

            cfg = obj.RunConfig.(run_fields{find(match_idx, 1)});
        end

        function pattern = get_run_pattern(obj, run_label)
            switch lower(run_label)
                case 'task'
                    pattern = obj.TaskPattern;
                case 'rest'
                    pattern = obj.RestPattern;
                otherwise
                    error('Unknown run label: %s', run_label);
            end
        end

        function field_name = get_behavioral_field(obj, run_label)
            switch lower(run_label)
                case 'task'
                    field_name = obj.TaskBehavioralField;
                case 'rest'
                    field_name = obj.RestBehavioralField;
                otherwise
                    error('Unknown run label: %s', run_label);
            end
        end

        function print_summary(obj)
            fprintf('\n=== Pipeline Configuration Summary ===\n');
            fprintf('Study Name       : %s\n', obj.StudyName);
            fprintf('Paradigm         : %s\n', obj.ExampleParadigm);
            fprintf('TR               : %.3f sec\n', obj.TR);
            fprintf('Atlas            : %s\n', obj.AtlasName);
            fprintf('Mask             : %s\n', obj.MaskName);
            fprintf('Denoising        : %s\n', obj.DenoisingMethod);
            fprintf('Participants     : %d\n', numel(obj.Participants));
            fprintf('Run Labels       : %s\n', strjoin(obj.RunLabels, ', '));
            fprintf('Methods          : %s\n', strjoin(obj.DefaultMethods, ', '));
            fprintf('Model Sets       : %s\n', strjoin(obj.ModelComparisonNames, ', '));
            fprintf('Analysis Mode    : %s\n', obj.AnalysisMode);
            fprintf('CV Scheme        : %s\n', obj.CVScheme);
            fprintf('======================================\n\n');
        end
    end
end

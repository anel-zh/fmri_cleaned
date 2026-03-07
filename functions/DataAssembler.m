classdef DataAssembler < handle
    % DATAASSEMBLER prepares model-ready datasets from extracted fMRI features
    %
    % Goals:
    %   - Load extracted task and rest feature files
    %   - Split data into train, validation, and test sets
    %   - Bin connectivity and apply variance normalization
    %   - Construct final model comparison datasets
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   assembler = DataAssembler(config);
    %   prepared_data = assembler.assemble_data();
    %
    % Inputs:
    %   - PipelineConfig object
    %   - Extracted DCC and HRF voxel feature files
    %   - Processed behavioral targets
    %
    % Outputs:
    %   - Prepared data structure used for model training
    %
    % Important notes:
    %   - X is stored as features x samples
    %   - one fold = one scan unit
    %   - task_rest models keep task + rest from the same scan unit in one fold
    %
    % Author: Anel Zhunussova

    properties
        Config      % PipelineConfig object
    end

    methods
        function obj = DataAssembler(config_obj)
            % Initialize assembler with pipeline configuration

            obj.Config = config_obj;
            obj.validate_config();
        end

        function prepared_data = assemble_data(obj, varargin)
            % Main entry point for data preparation

            p = inputParser;
            addParameter(p, 'FolderSuffix', '', @(x) ischar(x) || isstring(x));
            addParameter(p, 'NormType', 'wholebrain', @(x) ischar(x) || isstring(x));
            addParameter(p, 'SaveIntermediate', true, @islogical);
            parse(p, varargin{:});
            args = p.Results;

            % 1. Load all scan-unit data
            raw_data = obj.load_all_scan_units(args.FolderSuffix);

            % 2. Split by scan unit
            split_data = obj.divide_data_by_split(raw_data);

            % 3. Bin connectivity after split
            split_data_binned = obj.bin_connectivity_after_split(split_data);

            % 4. Variance normalization
            [vn_split_data, std_all] = obj.variance_normalize_after_split(split_data_binned, args.NormType);

            % 5. Build final comparison datasets
            models = obj.build_model_datasets(vn_split_data);

            prepared_data = struct();
            prepared_data.raw_data = raw_data;
            prepared_data.split_data = split_data;
            prepared_data.split_data_binned = split_data_binned;
            prepared_data.vn_split_data = vn_split_data;
            prepared_data.std_all = std_all;
            prepared_data.models = models;
            prepared_data.config = obj.Config;

            % 6. Optional saving
            if args.SaveIntermediate || obj.Config.SaveIntermediate
                save_dir = fullfile(obj.Config.ResultsDir, 'prepared_data');
                if ~exist(save_dir, 'dir')
                    mkdir(save_dir);
                end

                save(fullfile(save_dir, 'prepared_data_full.mat'), 'prepared_data', '-v7.3');
                save(fullfile(save_dir, 'split_data.mat'), 'split_data', '-v7.3');
                save(fullfile(save_dir, 'split_data_binned.mat'), 'split_data_binned', '-v7.3');
                save(fullfile(save_dir, 'vn_split_data.mat'), 'vn_split_data', '-v7.3');
                save(fullfile(save_dir, 'std_all.mat'), 'std_all', '-v7.3');
            end

            if obj.Config.Verbose
                obj.print_summary(models);
            end
        end
    end

    methods (Access = private)

        function validate_config(obj)
            % Checks that required configuration properties exist

            required_props = { ...
                'ResultsDir', 'Participants', 'DenoisingMethod', 'AtlasName', ...
                'TaskBehavioralField', 'RestBehavioralField', 'MetadataDir', ...
                'ModelSplit', 'RunConfig', 'ModelComparisonNames'};

            for i = 1:numel(required_props)
                if ~isprop(obj.Config, required_props{i})
                    error('Missing required config property: %s', required_props{i});
                end
            end
        end

        function raw_data = load_all_scan_units(obj, suffix)
            % Loads extracted data and behavioral targets for all scan units

            raw_data = struct();

            for i = 1:numel(obj.Config.Participants)
                scan_unit_id = obj.Config.Participants{i};

                if obj.Config.Verbose
                    fprintf('--- Loading scan unit: %s ---\n', scan_unit_id);
                end

                subject_dir = obj.get_subject_results_dir(scan_unit_id, suffix);
                safe_id = matlab.lang.makeValidName(scan_unit_id);

                raw_data.(safe_id) = struct();
                raw_data.(safe_id).scan_unit_id = scan_unit_id;

                % Imaging data
                raw_data.(safe_id).connectivity_task = obj.load_feature_file(subject_dir, 'dcc', 'task');
                raw_data.(safe_id).connectivity_rest = obj.load_feature_file(subject_dir, 'dcc', 'rest');

                raw_data.(safe_id).hrf_voxel_task = obj.load_feature_file(subject_dir, 'hrf_voxel', 'task');
                raw_data.(safe_id).hrf_voxel_rest = obj.load_feature_file(subject_dir, 'hrf_voxel', 'rest');

                % Behavioral data
                raw_data.(safe_id).behavioral_task = obj.load_behavioral_targets(scan_unit_id, 'task');
                raw_data.(safe_id).behavioral_rest = obj.load_behavioral_targets(scan_unit_id, 'rest');
            end
        end

        function subject_dir = get_subject_results_dir(obj, scan_unit_id, suffix)
            % Returns scan-unit-specific results directory

            suffix = char(string(suffix));

            if isempty(strtrim(suffix))
                folder_name = sprintf('%s_%s', ...
                    obj.Config.DenoisingMethod, obj.Config.AtlasName);
            else
                folder_name = sprintf('%s_%s_%s', ...
                    obj.Config.DenoisingMethod, obj.Config.AtlasName, suffix);
            end

            subject_dir = fullfile(obj.Config.ResultsDir, folder_name, scan_unit_id, 'data');
        end

        function feature_matrix = load_feature_file(obj, subject_dir, method, run_type)
            % Loads one feature matrix for one scan unit and one run

            feature_dir = obj.get_feature_directory(subject_dir, method);
            file_pattern = obj.get_feature_file_pattern(method, run_type);

            files = dir(fullfile(feature_dir, file_pattern));

            if isempty(files)
                warning('No %s file found for run type %s in %s', method, run_type, feature_dir);
                feature_matrix = [];
                return;
            elseif numel(files) > 1
                warning(['Multiple %s files found for run type %s in %s. ' ...
                         'Using the first one.'], method, run_type, feature_dir);
            end

            file_path = fullfile(files(1).folder, files(1).name);
            S = load(file_path);

            switch lower(method)
                case 'dcc'
                    if ~isfield(S, 'dcc_data')
                        error('Expected variable "dcc_data" missing in file: %s', file_path);
                    end
                    feature_matrix = S.dcc_data;     % edges x TR

                case 'hrf_voxel'
                    if ~isfield(S, 'beta_results')
                        error('Expected variable "beta_results" missing in file: %s', file_path);
                    end
                    feature_matrix = S.beta_results; % features x bins

                otherwise
                    error('Unsupported feature method: %s', method);
            end
        end

        function feature_dir = get_feature_directory(~, subject_dir, method)
            % Returns directory for a feature family

            switch lower(method)
                case 'dcc'
                    feature_dir = fullfile(subject_dir, 'DCC');

                case 'hrf_voxel'
                    feature_dir = fullfile(subject_dir, 'Activation');

                otherwise
                    error('Unknown feature method: %s', method);
            end
        end

        function file_pattern = get_feature_file_pattern(~, method, run_type)
            % Returns file pattern for one feature family and run type

            switch lower(method)
                case 'dcc'
                    file_pattern = sprintf('dcc_*%s*.mat', run_type);

                case 'hrf_voxel'
                    file_pattern = sprintf('hrf_voxel_*%s*.mat', run_type);

                otherwise
                    error('Unknown feature method: %s', method);
            end
        end

        function target_vector = load_behavioral_targets(obj, scan_unit_id, run_type)
            % Loads behavioral targets for one scan unit and one run

            metadata_file = fullfile(obj.Config.MetadataDir, [scan_unit_id '_behavioral.mat']);

            if ~exist(metadata_file, 'file')
                warning('Behavioral metadata file not found: %s', metadata_file);
                target_vector = [];
                return;
            end

            S = load(metadata_file);
            field_name = obj.Config.get_behavioral_field(run_type);

            if ~isfield(S, field_name)
                warning('Behavioral field %s not found in file: %s', field_name, metadata_file);
                target_vector = [];
                return;
            end

            target_vector = S.(field_name);
            if isrow(target_vector)
                target_vector = target_vector';
            end
        end

        function split_data = divide_data_by_split(obj, raw_data)
            % Divides scan units into train / valid / test sets

            split_data = struct();
            split_data.train = obj.init_split_struct();
            split_data.valid = obj.init_split_struct();
            split_data.test  = obj.init_split_struct();

            scan_unit_fields = fieldnames(raw_data);

            for i = 1:numel(scan_unit_fields)
                safe_id = scan_unit_fields{i};
                current = raw_data.(safe_id);
                scan_unit_id = current.scan_unit_id;

                if ismember(scan_unit_id, obj.Config.ModelSplit.train_ids)
                    split_name = 'train';
                elseif ismember(scan_unit_id, obj.Config.ModelSplit.valid_ids)
                    split_name = 'valid';
                elseif ismember(scan_unit_id, obj.Config.ModelSplit.test_ids)
                    split_name = 'test';
                else
                    warning('Scan unit %s was not assigned to any split. Skipping.', scan_unit_id);
                    continue;
                end

                split_data.(split_name).scan_unit_ids{end+1,1} = scan_unit_id;

                split_data.(split_name).connectivity_task{end+1,1} = current.connectivity_task;
                split_data.(split_name).connectivity_rest{end+1,1} = current.connectivity_rest;

                split_data.(split_name).hrf_voxel_task{end+1,1} = current.hrf_voxel_task;
                split_data.(split_name).hrf_voxel_rest{end+1,1} = current.hrf_voxel_rest;

                split_data.(split_name).behavioral_task{end+1,1} = current.behavioral_task;
                split_data.(split_name).behavioral_rest{end+1,1} = current.behavioral_rest;
            end
        end

        function split_struct = init_split_struct(~)
            % Initializes one split structure

            split_struct = struct();
            split_struct.scan_unit_ids = {};

            split_struct.connectivity_task = {};
            split_struct.connectivity_rest = {};

            split_struct.hrf_voxel_task = {};
            split_struct.hrf_voxel_rest = {};

            split_struct.behavioral_task = {};
            split_struct.behavioral_rest = {};
        end

        function split_data_binned = bin_connectivity_after_split(obj, split_data)
            % Bins connectivity after split and before variance normalization
            %
            % Unified modeling rule:
            %   - hrf_voxel enters modeling already binned
            %   - connectivity is first binned here
            %   - both modalities are variance-normalized only after binning

            split_data_binned = split_data;

            fields = {'train', 'valid', 'test'};

            for f = 1:numel(fields)
                field_name = fields{f};

                for i = 1:numel(split_data.(field_name).scan_unit_ids)
                    connectivity_task = split_data.(field_name).connectivity_task{i};
                    connectivity_rest = split_data.(field_name).connectivity_rest{i};

                    binned_connectivity = obj.bin_connectivity(connectivity_task, connectivity_rest);

                    split_data_binned.(field_name).connectivity_task{i} = binned_connectivity.connectivity_task;
                    split_data_binned.(field_name).connectivity_rest{i} = binned_connectivity.connectivity_rest;
                end
            end
        end

        function [vn_split_data, std_all] = variance_normalize_after_split(~, split_data_binned, normType)
            % Applies variance normalization after binning
            %
            % Current implementation:
            %   - standard deviation estimated from training split only
            %   - same training-derived std applied to valid/test
            %   - normalization performed feature-wise

            if nargin < 3
                normType = 'wholebrain';
            end

            vn_split_data = split_data_binned;
            std_all = struct();
            std_all.normType = char(string(normType));

            % HRF voxel normalization
            train_hrf_task = cat(2, split_data_binned.train.hrf_voxel_task{:});
            train_hrf_rest = cat(2, split_data_binned.train.hrf_voxel_rest{:});
            train_hrf_all = [train_hrf_task, train_hrf_rest];

            std_all.hrf_voxel = std(train_hrf_all, 0, 2, 'omitnan');
            std_all.hrf_voxel(std_all.hrf_voxel == 0) = 1;

            % DCC normalization
            train_dcc_task = cat(2, split_data_binned.train.connectivity_task{:});
            train_dcc_rest = cat(2, split_data_binned.train.connectivity_rest{:});
            train_dcc_all = [train_dcc_task, train_dcc_rest];

            std_all.connectivity = std(train_dcc_all, 0, 2, 'omitnan');
            std_all.connectivity(std_all.connectivity == 0) = 1;

            fields = {'train', 'valid', 'test'};
            for f = 1:numel(fields)
                field_name = fields{f};

                for i = 1:numel(split_data_binned.(field_name).scan_unit_ids)
                    % HRF voxel
                    vn_split_data.(field_name).hrf_voxel_task{i} = ...
                        split_data_binned.(field_name).hrf_voxel_task{i} ./ std_all.hrf_voxel;

                    vn_split_data.(field_name).hrf_voxel_rest{i} = ...
                        split_data_binned.(field_name).hrf_voxel_rest{i} ./ std_all.hrf_voxel;

                    % DCC
                    vn_split_data.(field_name).connectivity_task{i} = ...
                        split_data_binned.(field_name).connectivity_task{i} ./ std_all.connectivity;

                    vn_split_data.(field_name).connectivity_rest{i} = ...
                        split_data_binned.(field_name).connectivity_rest{i} ./ std_all.connectivity;
                end
            end
        end

        function binned = bin_connectivity(obj, connectivity_task, connectivity_rest)
            % Bins TR-level connectivity into window-level connectivity
            %
            % Input:
            %   connectivity_task : edges x TR
            %   connectivity_rest : edges x TR
            %
            % Output:
            %   connectivity_task : edges x n_task_bins
            %   connectivity_rest : edges x n_rest_bins

            task_cfg = obj.Config.get_run_config('task');
            rest_cfg = obj.Config.get_run_config('rest');

            binned = struct();
            binned.connectivity_task = obj.bin_one_run( ...
                connectivity_task, ...
                task_cfg.WindowSizeTR, ...
                task_cfg.NumBins, ...
                task_cfg.StartTR, ...
                task_cfg.MaxTR, ...
                task_cfg.RemoveStimulusDelivery, ...
                task_cfg.StimulusDeliveryEndSec);

            binned.connectivity_rest = obj.bin_one_run( ...
                connectivity_rest, ...
                rest_cfg.WindowSizeTR, ...
                rest_cfg.NumBins, ...
                rest_cfg.StartTR, ...
                rest_cfg.MaxTR, ...
                false, ...
                []);
        end

        function binned_matrix = bin_one_run(obj, run_matrix, window_size_TR, num_bins, start_tr, max_tr, remove_stimulus, stimulus_end_sec)
            % Bins one run by averaging TRs within each bin
            %
            % Input:
            %   run_matrix : features x TR
            %
            % Output:
            %   binned_matrix : features x bins

            if isempty(run_matrix)
                binned_matrix = [];
                return;
            end

            total_TR = size(run_matrix, 2);
            end_tr = min(total_TR, max_tr);

            if remove_stimulus
                stimulus_end_tr = floor(stimulus_end_sec / obj.Config.TR);
                start_tr = max(start_tr, stimulus_end_tr + 1);
            end

            available_TRs = end_tr - start_tr + 1;
            max_possible_bins = floor(available_TRs / window_size_TR);
            num_bins = min(num_bins, max_possible_bins);

            if num_bins < 1
                binned_matrix = [];
                return;
            end

            binned_matrix = zeros(size(run_matrix, 1), num_bins);

            for b = 1:num_bins
                tr_start = start_tr + (b - 1) * window_size_TR;
                tr_end = tr_start + window_size_TR - 1;

                current_bin = run_matrix(:, tr_start:tr_end);
                binned_matrix(:, b) = mean(current_bin, 2, 'omitnan');
            end
        end

        function models = build_model_datasets(obj, vn_split_data)
            % Builds 6 final comparison datasets from split data

            model_names = obj.Config.ModelComparisonNames;
            models = struct();

            for i = 1:numel(model_names)
                models.(model_names{i}) = obj.initialize_model_struct(model_names{i});
            end

            fields = {'train', 'valid', 'test'};

            for f = 1:numel(fields)
                split_name = fields{f};
                current_split = vn_split_data.(split_name);

                for i = 1:numel(current_split.scan_unit_ids)
                    scan_unit_id = current_split.scan_unit_ids{i};

                    dcc_task = current_split.connectivity_task{i};
                    dcc_rest = current_split.connectivity_rest{i};

                    hrf_voxel_task = current_split.hrf_voxel_task{i};
                    hrf_voxel_rest = current_split.hrf_voxel_rest{i};

                    task_y = current_split.behavioral_task{i};
                    rest_y = current_split.behavioral_rest{i};

                    scan_models = obj.build_scan_unit_models( ...
                        scan_unit_id, ...
                        dcc_task, dcc_rest, ...
                        hrf_voxel_task, hrf_voxel_rest, ...
                        task_y, rest_y);

                    models = obj.append_scan_unit_models(models, scan_models);
                end
            end
        end

        function model_struct = initialize_model_struct(~, model_name)
            % Creates empty storage structure for one model dataset

            model_struct = struct();
            model_struct.model_name = model_name;
            model_struct.X = [];
            model_struct.y = [];
            model_struct.whfolds = [];
            model_struct.meta = struct( ...
                'scan_unit_id', {{}}, ...
                'run_condition', {{}}, ...
                'bin_labels', {{}}, ...
                'feature_family', {{}});
        end

        function scan_models = build_scan_unit_models(~, scan_unit_id, ...
                dcc_task, dcc_rest, hrf_voxel_task, hrf_voxel_rest, task_y, rest_y)
            % Builds the six final comparison datasets for one scan unit (one participant / one session)

            scan_models = struct();

            % 1. dcc_task
            scan_models.dcc_task = [];
            if ~isempty(dcc_task) && ~isempty(task_y)
                scan_models.dcc_task = struct();
                scan_models.dcc_task.X = dcc_task;
                scan_models.dcc_task.y = match_target_length(task_y, size(dcc_task, 2), scan_unit_id, 'task', 'dcc');
                scan_models.dcc_task.whfolds = ones(1, size(dcc_task, 2));
                scan_models.dcc_task.meta = build_meta(scan_unit_id, 'task', 'dcc', size(dcc_task, 2), 'task');
            end

            % 2. dcc_task_rest
            scan_models.dcc_task_rest = [];
            if ~isempty(dcc_task) && ~isempty(dcc_rest) && ~isempty(task_y) && ~isempty(rest_y)
                X_tmp = [dcc_task, dcc_rest];
                y_tmp = [match_target_length(task_y, size(dcc_task, 2), scan_unit_id, 'task', 'dcc'); ...
                         match_target_length(rest_y, size(dcc_rest, 2), scan_unit_id, 'rest', 'dcc')];

                scan_models.dcc_task_rest = struct();
                scan_models.dcc_task_rest.X = X_tmp;
                scan_models.dcc_task_rest.y = y_tmp;
                scan_models.dcc_task_rest.whfolds = ones(1, size(X_tmp, 2));
                scan_models.dcc_task_rest.meta = build_meta(scan_unit_id, 'task_rest', 'dcc', ...
                    size(X_tmp, 2), [repmat({'task'}, size(dcc_task, 2), 1); ...
                                     repmat({'rest'}, size(dcc_rest, 2), 1)]);
            end

            % 3. hrf_voxel_task
            scan_models.hrf_voxel_task = [];
            if ~isempty(hrf_voxel_task) && ~isempty(task_y)
                scan_models.hrf_voxel_task = struct();
                scan_models.hrf_voxel_task.X = hrf_voxel_task;
                scan_models.hrf_voxel_task.y = match_target_length(task_y, size(hrf_voxel_task, 2), scan_unit_id, 'task', 'hrf_voxel');
                scan_models.hrf_voxel_task.whfolds = ones(1, size(hrf_voxel_task, 2));
                scan_models.hrf_voxel_task.meta = build_meta(scan_unit_id, 'task', 'hrf_voxel', size(hrf_voxel_task, 2), 'task');
            end

            % 4. hrf_voxel_task_rest
            scan_models.hrf_voxel_task_rest = [];
            if ~isempty(hrf_voxel_task) && ~isempty(hrf_voxel_rest) && ~isempty(task_y) && ~isempty(rest_y)
                X_tmp = [hrf_voxel_task, hrf_voxel_rest];
                y_tmp = [match_target_length(task_y, size(hrf_voxel_task, 2), scan_unit_id, 'task', 'hrf_voxel'); ...
                         match_target_length(rest_y, size(hrf_voxel_rest, 2), scan_unit_id, 'rest', 'hrf_voxel')];

                scan_models.hrf_voxel_task_rest = struct();
                scan_models.hrf_voxel_task_rest.X = X_tmp;
                scan_models.hrf_voxel_task_rest.y = y_tmp;
                scan_models.hrf_voxel_task_rest.whfolds = ones(1, size(X_tmp, 2));
                scan_models.hrf_voxel_task_rest.meta = build_meta(scan_unit_id, 'task_rest', 'hrf_voxel', ...
                    size(X_tmp, 2), [repmat({'task'}, size(hrf_voxel_task, 2), 1); ...
                                     repmat({'rest'}, size(hrf_voxel_rest, 2), 1)]);
            end

            % 5. combined_task
            scan_models.combined_task = [];
            if ~isempty(dcc_task) && ~isempty(hrf_voxel_task) && ~isempty(task_y)
                assert(size(dcc_task, 2) == size(hrf_voxel_task, 2), ...
                    'Combined task model requires matching task bin counts.');

                X_tmp = [dcc_task; hrf_voxel_task];
                y_tmp = match_target_length(task_y, size(X_tmp, 2), scan_unit_id, 'task', 'combined');

                scan_models.combined_task = struct();
                scan_models.combined_task.X = X_tmp;
                scan_models.combined_task.y = y_tmp;
                scan_models.combined_task.whfolds = ones(1, size(X_tmp, 2));
                scan_models.combined_task.meta = build_meta(scan_unit_id, 'task', 'combined', size(X_tmp, 2), 'task');
            end

            % 6. combined_task_rest
            scan_models.combined_task_rest = [];
            if ~isempty(dcc_task) && ~isempty(dcc_rest) && ...
               ~isempty(hrf_voxel_task) && ~isempty(hrf_voxel_rest) && ...
               ~isempty(task_y) && ~isempty(rest_y)

                assert(size(dcc_task, 2) == size(hrf_voxel_task, 2), ...
                    'Combined task model requires matching task bin counts.');
                assert(size(dcc_rest, 2) == size(hrf_voxel_rest, 2), ...
                    'Combined rest model requires matching rest bin counts.');

                dcc_tmp = [dcc_task, dcc_rest];
                hrf_tmp = [hrf_voxel_task, hrf_voxel_rest];
                X_tmp = [dcc_tmp; hrf_tmp];

                y_tmp = [match_target_length(task_y, size(dcc_task, 2), scan_unit_id, 'task', 'combined'); ...
                         match_target_length(rest_y, size(dcc_rest, 2), scan_unit_id, 'rest', 'combined')];

                scan_models.combined_task_rest = struct();
                scan_models.combined_task_rest.X = X_tmp;
                scan_models.combined_task_rest.y = y_tmp;
                scan_models.combined_task_rest.whfolds = ones(1, size(X_tmp, 2));
                scan_models.combined_task_rest.meta = build_meta(scan_unit_id, 'task_rest', 'combined', ...
                    size(X_tmp, 2), [repmat({'task'}, size(dcc_task, 2), 1); ...
                                     repmat({'rest'}, size(dcc_rest, 2), 1)]);
            end

            function y_out = match_target_length(y_in, expected_len, current_scan_id, current_run, current_method)
                if numel(y_in) ~= expected_len
                    error(['Behavioral target length does not match extracted bin count for ' ...
                           '%s | run: %s | method: %s | expected: %d | found: %d'], ...
                           current_scan_id, current_run, current_method, expected_len, numel(y_in));
                end
                y_out = y_in(:);
            end

            function meta_out = build_meta(current_scan_id, run_condition, feature_family, n_samples, run_labels)
                if ischar(run_labels) || isstring(run_labels)
                    run_labels = repmat({char(string(run_labels))}, n_samples, 1);
                end

                meta_out = struct();
                meta_out.scan_unit_id = repmat({current_scan_id}, n_samples, 1);
                meta_out.run_condition = repmat({run_condition}, n_samples, 1);
                meta_out.bin_labels = run_labels;
                meta_out.feature_family = repmat({feature_family}, n_samples, 1);
            end
        end

        function models = append_scan_unit_models(~, models, scan_models)
            % Appends one scan unit's data to the global model datasets

            model_names = fieldnames(models);

            for i = 1:numel(model_names)
                model_name = model_names{i};

                if isempty(scan_models.(model_name))
                    continue;
                end

                current_model = models.(model_name);
                new_model = scan_models.(model_name);

                current_fold_id = get_next_fold_id(current_model.whfolds);
                new_model.whfolds = current_fold_id * ones(1, size(new_model.X, 2));

                current_model.X = [current_model.X, new_model.X];
                current_model.y = [current_model.y; new_model.y];
                current_model.whfolds = [current_model.whfolds, new_model.whfolds];

                current_model.meta.scan_unit_id = [current_model.meta.scan_unit_id; new_model.meta.scan_unit_id];
                current_model.meta.run_condition = [current_model.meta.run_condition; new_model.meta.run_condition];
                current_model.meta.bin_labels = [current_model.meta.bin_labels; new_model.meta.bin_labels];
                current_model.meta.feature_family = [current_model.meta.feature_family; new_model.meta.feature_family];

                models.(model_name) = current_model;
            end

            function next_id = get_next_fold_id(existing_whfolds)
                if isempty(existing_whfolds)
                    next_id = 1;
                else
                    next_id = max(existing_whfolds) + 1;
                end
            end
        end

        function print_summary(~, models)
            % Prints brief summary of final assembled datasets

            fprintf('\n=== Data Assembly Summary ===\n');

            model_names = fieldnames(models);
            for i = 1:numel(model_names)
                model_name = model_names{i};
                model_obj = models.(model_name);

                fprintf('%s\n', model_name);
                fprintf('  X shape   : %d x %d\n', size(model_obj.X, 1), size(model_obj.X, 2));
                fprintf('  y length  : %d\n', numel(model_obj.y));
                fprintf('  n_folds   : %d\n', numel(unique(model_obj.whfolds)));
            end

            fprintf('=============================\n\n');
        end
    end
end

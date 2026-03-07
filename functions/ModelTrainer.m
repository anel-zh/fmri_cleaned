classdef ModelTrainer < handle
    % MODELTRAINER performs model training, validation, and testing for fMRI-based prediction models
    %
    % Goals:
    %   - Train predictive models using cross-validation
    %   - Select the optimal number of components via validation
    %   - Evaluate final model performance on the test dataset
    %   - Save model weights, predictions, and evaluation metrics
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   trainer = ModelTrainer(config);
    %   results = trainer.train_all_models(prepared_data);
    %
    % Inputs:
    %   - PipelineConfig object
    %   - Prepared model datasets
    %
    % Outputs:
    %   - Results structure containing:
    %         training results
    %         validation results
    %         testing results
    %         model weights
    %
    % Author: Anel Zhunussova

    properties
        Config      % PipelineConfig object
    end

    methods
        function obj = ModelTrainer(config_obj)
            % Initialize trainer with pipeline configuration
            obj.Config = config_obj;
            obj.validate_config();
        end

        function results_structure = train_all_models(obj, assembled_data, varargin)
            % Main loop over all model comparison datasets

            p = inputParser;
            addParameter(p, 'SaveFolderName', 'model_training', @(x) ischar(x) || isstring(x));
            addParameter(p, 'PCRange', [], @isnumeric);
            addParameter(p, 'PerformanceMetric', 'mean_r', @(x) ischar(x) || isstring(x));
            addParameter(p, 'SaveResults', true, @islogical);
            parse(p, varargin{:});
            args = p.Results;

            save_dir = fullfile(obj.Config.ResultsDir, 'model_training', char(string(args.SaveFolderName)));
            if args.SaveResults && ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end

            results_structure = obj.initialize_results_structure(args);

            if isprop(obj.Config, 'ModelComparisonNames') && ~isempty(obj.Config.ModelComparisonNames)
                model_names = obj.Config.ModelComparisonNames;
            else
                model_names = fieldnames(assembled_data.models);
            end

            for m = 1:numel(model_names)
                model_name = model_names{m};

                if ~isfield(assembled_data.models, model_name)
                    warning('Model dataset %s was not found in assembled_data.models. Skipping.', model_name);
                    continue;
                end

                dataset = assembled_data.models.(model_name);

                if isempty(dataset.X)
                    warning('Skipping empty model dataset: %s', model_name);
                    continue;
                end

                obj.print_model_header(model_name);

                % 1. Split data
                split_data = obj.build_split_datasets(dataset);

                if obj.is_invalid_split(split_data, model_name)
                    continue;
                end

                % 2. Define PC range
                if isempty(args.PCRange)
                    PC_range = obj.get_pc_range(split_data.train);
                else
                    PC_range = args.PCRange;
                end

                % 3. Training
                [training_result, stats_table, model_weights_table] = ...
                    obj.training(split_data.train, PC_range);

                % 4. Validation
                validation_result = obj.validation( ...
                    split_data.valid, model_weights_table, args.PerformanceMetric);

                % 5. Best PC
                best_PC = validation_result.best_PC;

                % 6. Testing
                testing_result = obj.testing( ...
                    split_data.test, model_weights_table, best_PC);

                % 7. Save into results structure
                results_structure.training.(model_name) = training_result;
                results_structure.training.(model_name).stats_table = stats_table;

                results_structure.validation.(model_name) = validation_result;
                results_structure.testing.(model_name) = testing_result;
                results_structure.model_weights.(model_name) = model_weights_table;
                results_structure.best_PC.(model_name) = best_PC;
                results_structure.model_info.(model_name) = obj.get_model_info(dataset, split_data, PC_range);

                % 8. Save per-model outputs
                if args.SaveResults
                    obj.save_model_outputs(save_dir, model_name, ...
                        training_result, validation_result, testing_result, ...
                        model_weights_table, stats_table, best_PC);
                end
            end

            if args.SaveResults
                save(fullfile(save_dir, 'results_structure_full.mat'), 'results_structure', '-v7.3');
            end
        end
    end

    methods (Access = private)

        function validate_config(obj)
            % Checks that required configuration fields exist

            required_props = {'ResultsDir', 'Verbose', 'ModelSplit'};

            for i = 1:numel(required_props)
                if ~isprop(obj.Config, required_props{i})
                    error('Missing required config property: %s', required_props{i});
                end
            end

            required_split_fields = {'train_ids', 'valid_ids', 'test_ids'};
            for i = 1:numel(required_split_fields)
                if ~isfield(obj.Config.ModelSplit, required_split_fields{i})
                    error('Config.ModelSplit is missing field: %s', required_split_fields{i});
                end
                if ~iscell(obj.Config.ModelSplit.(required_split_fields{i}))
                    error('Config.ModelSplit.%s must be a cell array.', required_split_fields{i});
                end
            end
        end

        function results_structure = initialize_results_structure(~, args)
            % Creates top-level results storage

            results_structure = struct();
            results_structure.training = struct();
            results_structure.validation = struct();
            results_structure.testing = struct();
            results_structure.model_weights = struct();
            results_structure.best_PC = struct();
            results_structure.model_info = struct();
            results_structure.run_settings = struct( ...
                'PCRange', args.PCRange, ...
                'PerformanceMetric', char(string(args.PerformanceMetric)), ...
                'SaveResults', args.SaveResults);
        end

        function print_model_header(obj, model_name)
            % Prints model header for readability

            if obj.Config.Verbose
                fprintf('\n====================================================\n');
                fprintf('Running model: %s\n', model_name);
                fprintf('====================================================\n');
            end
        end

        function split_data = build_split_datasets(obj, dataset)
            % Builds train / valid / test subsets from scan-unit IDs

            split_data = struct();
            split_data.train = obj.get_split_dataset(dataset, obj.Config.ModelSplit.train_ids);
            split_data.valid = obj.get_split_dataset(dataset, obj.Config.ModelSplit.valid_ids);
            split_data.test  = obj.get_split_dataset(dataset, obj.Config.ModelSplit.test_ids);
        end

        function tf = is_invalid_split(~, split_data, model_name)
            % Checks whether required splits are empty

            tf = false;

            if isempty(split_data.train.X)
                warning('Training split is empty for model: %s. Skipping.', model_name);
                tf = true;
                return;
            end

            if isempty(split_data.valid.X)
                warning('Validation split is empty for model: %s. Skipping.', model_name);
                tf = true;
                return;
            end

            if isempty(split_data.test.X)
                warning('Testing split is empty for model: %s. Skipping.', model_name);
                tf = true;
            end
        end

        function subset = get_split_dataset(~, dataset, selected_ids)
            % Returns subset of a model dataset for selected scan-unit IDs

            scan_ids = dataset.meta.scan_unit_id;
            keep_mask = ismember(scan_ids, selected_ids);

            subset = struct();
            subset.model_name = dataset.model_name;
            subset.X = dataset.X(:, keep_mask);
            subset.y = dataset.y(keep_mask);
            subset.whfolds = dataset.whfolds(keep_mask);

            subset.meta = struct();
            subset.meta.scan_unit_id = dataset.meta.scan_unit_id(keep_mask);
            subset.meta.run_condition = dataset.meta.run_condition(keep_mask);
            subset.meta.bin_labels = dataset.meta.bin_labels(keep_mask);
            subset.meta.feature_family = dataset.meta.feature_family(keep_mask);
        end

        function PC_range = get_pc_range(~, train_set)
            % Determines default PC range from training sample count

            max_pc = min(size(train_set.X, 2) - 1, 20);

            if max_pc < 1
                error('Training set is too small to define a valid PC range.');
            end

            PC_range = 1:max_pc;
        end

        function [main_table_result, stats_table, model_weights_table] = training(obj, train_set, PC_range)
            % Training across a PC range using CV-PCR

            main_table_result = table();
            stats_table = table();
            model_weights_table = table();

            dat = fmri_data;
            dat.dat = train_set.X;
            dat.Y = train_set.y;

            whfolds = train_set.whfolds(:);

            for PC = PC_range
                if obj.Config.Verbose
                    fprintf('Training PC %d / %d | %s\n', PC, max(PC_range), train_set.model_name);
                end

                [~, stats, optout] = predict(dat, ...
                    'algorithm_name', 'cv_pcr', ...
                    'nfolds', whfolds, ...
                    'numcomponents', PC, ...
                    'verbose', 1, ...
                    'error_type', 'mse');

                performance = obj.get_performance(stats.Y, stats.yfit, whfolds);
                ratings_all = obj.get_shaped_outputs(train_set, stats.Y, stats.yfit);

                temp_table_result = table( ...
                    PC, {ratings_all}, {performance.r}, performance.mean_r, ...
                    performance.corr_y, performance.rsquare, performance.mse, ...
                    'VariableNames', {'PC', 'ratings_all', 'r', 'mean_r', 'corr_y', 'rsquare', 'mse'});

                main_table_result = [main_table_result; temp_table_result];

                stats_row = table(PC, {stats}, {optout}, ...
                    'VariableNames', {'PC', 'stats', 'optout'});
                stats_table = [stats_table; stats_row];

                [model_weights, intercept] = obj.extract_model_weights(optout);

                weights_row = table( ...
                    PC, {model_weights}, intercept, ...
                    'VariableNames', {'PC', 'model_weights', 'intercept'});

                model_weights_table = [model_weights_table; weights_row];
            end
        end

        function validation_result = validation(obj, valid_set, model_weights_table, performance_metric)
            % Applies training weights to validation data for each PC

            performance_table = table();

            for i = 1:height(model_weights_table)
                PC = model_weights_table.PC(i);
                model_weights = model_weights_table.model_weights{i};
                intercept = model_weights_table.intercept(i);

                if obj.Config.Verbose
                    fprintf('Validation PC %d | %s\n', PC, valid_set.model_name);
                end

                yfit = obj.apply_model_weights(valid_set.X, model_weights, intercept);
                performance = obj.get_performance(valid_set.y, yfit, valid_set.whfolds);
                ratings_all = obj.get_shaped_outputs(valid_set, valid_set.y, yfit);

                temp_row = table( ...
                    PC, {ratings_all}, {performance.r}, performance.mean_r, ...
                    performance.corr_y, performance.rsquare, performance.mse, ...
                    'VariableNames', {'PC', 'ratings_all', 'r', 'mean_r', 'corr_y', 'rsquare', 'mse'});

                performance_table = [performance_table; temp_row];
            end

            best_PC = obj.choose_best_pc(performance_table, performance_metric);

            validation_result = struct();
            validation_result.performance_table = performance_table;
            validation_result.best_PC = best_PC;
            validation_result.selection_metric = char(string(performance_metric));
        end

        function testing_result = testing(obj, test_set, model_weights_table, best_PC)
            % Applies best validation model to held-out test data

            row_idx = model_weights_table.PC == best_PC;
            if ~any(row_idx)
                error('Best PC %d was not found in model_weights_table.', best_PC);
            end

            model_weights = model_weights_table.model_weights{row_idx};
            intercept = model_weights_table.intercept(row_idx);

            if obj.Config.Verbose
                fprintf('Testing best PC %d | %s\n', best_PC, test_set.model_name);
            end

            yfit = obj.apply_model_weights(test_set.X, model_weights, intercept);
            performance = obj.get_performance(test_set.y, yfit, test_set.whfolds);
            ratings_all = obj.get_shaped_outputs(test_set, test_set.y, yfit);

            testing_result = struct();
            testing_result.best_PC = best_PC;
            testing_result.model_weights = model_weights;
            testing_result.intercept = intercept;

            testing_result.ratings_all = ratings_all;
            testing_result.r = performance.r;
            testing_result.mean_r = performance.mean_r;
            testing_result.corr_y = performance.corr_y;
            testing_result.rsquare = performance.rsquare;
            testing_result.mse = performance.mse;

            testing_result.y = test_set.y;
            testing_result.yfit = yfit;
        end

        function [model_weights, intercept] = extract_model_weights(~, optout)
            % Extracts learned model weights and intercept from Canlab predict output

            if ~iscell(optout) || numel(optout) < 2
                error('optout does not match expected Canlab predict output format.');
            end

            model_weights = optout{1};
            intercept = optout{2};

            if ~isnumeric(model_weights)
                error('optout{1} is not numeric model weights.');
            end

            if iscell(intercept)
                intercept = intercept{1};
            end

            if ~isscalar(intercept) || ~isnumeric(intercept)
                error('optout{2} is not a numeric scalar intercept.');
            end
        end

        function yfit = apply_model_weights(~, X, model_weights, intercept)
            % Applies learned model weights to a features x samples matrix

            if size(X, 1) ~= numel(model_weights)
                error(['Feature dimension mismatch when applying model weights. ' ...
                       'X has %d features, model_weights has %d elements.'], ...
                       size(X, 1), numel(model_weights));
            end

            yfit = (model_weights(:)' * X)' + intercept;
        end

        function best_PC = choose_best_pc(~, performance_table, metric_name)
            % Selects best PC using validation metric

            metric_name = lower(char(string(metric_name)));

            switch metric_name
                case 'mean_r'
                    [~, idx] = max(performance_table.mean_r);

                case 'corr_y'
                    [~, idx] = max(performance_table.corr_y);

                case 'rsquare'
                    [~, idx] = max(performance_table.rsquare);

                case 'mse'
                    [~, idx] = min(performance_table.mse);

                otherwise
                    error('Unsupported performance metric: %s', metric_name);
            end

            best_PC = performance_table.PC(idx);
        end

        function performance = get_performance(~, y, yfit, whfolds)
            % Computes performance

            y = y(:);
            yfit = yfit(:);
            whfolds = whfolds(:);

            unique_folds = unique(whfolds);
            r = nan(numel(unique_folds), 1);

            for i = 1:numel(unique_folds)
                idx = whfolds == unique_folds(i);

                if sum(idx) > 1
                    r(i) = corr(y(idx), yfit(idx), 'rows', 'complete');
                end
            end

            corr_y = corr(y, yfit, 'rows', 'complete');

            performance = struct();
            performance.r = r;
            performance.mean_r = mean(r, 'omitnan');
            performance.corr_y = corr_y;
            performance.rsquare = corr_y^2;
            performance.mse = mean((y - yfit).^2, 'omitnan');
        end

        function ratings_all = get_shaped_outputs(~, dataset, y, yfit)
            % Organizes outputs in a structure

            y = y(:);
            yfit = yfit(:);

            run_condition = dataset.meta.run_condition{1};
            scan_ids = dataset.meta.scan_unit_id;
            unique_scans = unique(scan_ids, 'stable');

            ratings_all = struct();
            ratings_all.model_performance = struct();
            ratings_all.plot = struct();

            ratings_all.model_performance.(run_condition).y = y;
            ratings_all.model_performance.(run_condition).yfit = yfit;

            n_scans = numel(unique_scans);

            first_scan_idx = strcmp(scan_ids, unique_scans{1});
            n_bins = sum(first_scan_idx);

            y_mat = nan(n_scans, n_bins);
            yfit_mat = nan(n_scans, n_bins);

            for i = 1:n_scans
                idx = strcmp(scan_ids, unique_scans{i});

                y_i = y(idx);
                yfit_i = yfit(idx);

                if numel(y_i) ~= n_bins
                    error('Inconsistent number of bins across scan units in get_shaped_outputs.');
                end

                y_mat(i, :) = y_i(:)';
                yfit_mat(i, :) = yfit_i(:)';
            end

            ratings_all.plot.(run_condition).y = y_mat;
            ratings_all.plot.(run_condition).yfit = yfit_mat;
        end

        function model_info = get_model_info(~, dataset, split_data, PC_range)
            % Saves convenient model metadata for later inspection

            model_info = struct();
            model_info.model_name = dataset.model_name;
            model_info.n_features = size(dataset.X, 1);
            model_info.n_samples = size(dataset.X, 2);
            model_info.PC_range = PC_range;

            model_info.n_train_samples = size(split_data.train.X, 2);
            model_info.n_valid_samples = size(split_data.valid.X, 2);
            model_info.n_test_samples = size(split_data.test.X, 2);

            model_info.n_train_folds = numel(unique(split_data.train.whfolds));
            model_info.n_valid_folds = numel(unique(split_data.valid.whfolds));
            model_info.n_test_folds = numel(unique(split_data.test.whfolds));
        end

        function save_model_outputs(~, save_dir, model_name, ...
                training_result, validation_result, testing_result, ...
                model_weights_table, stats_table, best_PC)
            % Saves per-model outputs into dedicated folder

            model_save_dir = fullfile(save_dir, model_name);
            if ~exist(model_save_dir, 'dir')
                mkdir(model_save_dir);
            end

            training_output = training_result;
            validation_output = validation_result;
            testing_output = testing_result;
            weights_output = model_weights_table;
            stats_output = stats_table;
            best_PC_output = best_PC;

            save(fullfile(model_save_dir, 'training_result.mat'), 'training_output', '-v7.3');
            save(fullfile(model_save_dir, 'validation_result.mat'), 'validation_output', '-v7.3');
            save(fullfile(model_save_dir, 'testing_result.mat'), 'testing_output', '-v7.3');
            save(fullfile(model_save_dir, 'model_weights_table.mat'), 'weights_output', '-v7.3');
            save(fullfile(model_save_dir, 'stats_table.mat'), 'stats_output', '-v7.3');
            save(fullfile(model_save_dir, 'best_PC.mat'), 'best_PC_output', '-v7.3');
        end
    end
end

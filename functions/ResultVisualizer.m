classdef ResultVisualizer < handle
    % RESULTVISUALIZER generates evaluation plots and summaries for fMRI prediction model results
    %
    % Goals:
    %   - Visualize validation-based model selection
    %   - Plot observed versus predicted outcomes
    %   - Inspect model weights and prediction patterns
    %   - Summarize model performance across comparison models
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   visualizer = ResultVisualizer(config);
    %   visualizer.plot_results(results);
    %
    % Inputs:
    %   - PipelineConfig object
    %   - Model results structure
    %
    % Outputs:
    %   - Performance figures and diagnostic plots
    %
    % Author: Anel Zhunussova
    
    properties
        Config      % PipelineConfig object
    end

    methods
        function obj = ResultVisualizer(config_obj)
            % Initialize visualizer with pipeline configuration

            obj.Config = config_obj;
        end

        function plot_pc_selection(obj, results_structure, model_name, varargin)
            % Plots validation performance across PCs for one model

            p = inputParser;
            addParameter(p, 'Metric', 'mean_r', @(x) ischar(x) || isstring(x));
            addParameter(p, 'ShowBestPC', true, @islogical);
            parse(p, varargin{:});
            args = p.Results;

            validation_result = obj.get_model_result(results_structure, 'validation', model_name);
            performance_table = validation_result.performance_table;
            metric_name = char(string(args.Metric));

            obj.validate_metric(performance_table, metric_name, 'validation', model_name);

            yvals = performance_table.(metric_name);
            if iscell(yvals)
                error('Metric %s is not scalar and cannot be plotted directly.', metric_name);
            end

            figure;
            plot(performance_table.PC, yvals, '-o', 'LineWidth', 1.5);
            xlabel('Number of Principal Components');
            ylabel(metric_name);
            title(sprintf('%s | Validation %s across PCs', model_name, metric_name), 'Interpreter', 'none');
            grid on;

            if args.ShowBestPC && isfield(validation_result, 'best_PC')
                hold on;
                xline(validation_result.best_PC, '--', 'Best PC', 'LineWidth', 1.2);
                hold off;
            end
        end

        function plot_observed_vs_predicted(obj, results_structure, model_name, split_name, varargin)
            % Plots observed vs predicted values for one split and one model

            p = inputParser;
            addParameter(p, 'RunCondition', '', @(x) ischar(x) || isstring(x));
            addParameter(p, 'AddReferenceLine', true, @islogical);
            parse(p, varargin{:});
            args = p.Results;

            split_result = obj.get_model_result(results_structure, split_name, model_name);
            ratings_all = obj.extract_ratings_all(split_result, split_name);

            available_conditions = fieldnames(ratings_all.model_performance);
            run_condition = obj.resolve_run_condition(args.RunCondition, available_conditions);

            y = ratings_all.model_performance.(run_condition).y;
            yfit = ratings_all.model_performance.(run_condition).yfit;

            figure;
            scatter(y, yfit, 20, 'filled');
            xlabel('Observed');
            ylabel('Predicted');
            title(sprintf('%s | %s | %s observed vs predicted', ...
                model_name, obj.pretty_split_name(split_name), run_condition), 'Interpreter', 'none');
            grid on;

            if args.AddReferenceLine
                hold on;
                lims = [min([y; yfit]), max([y; yfit])];
                plot(lims, lims, '--', 'LineWidth', 1.2);
                hold off;
            end
        end

        function plot_grouped_predictions(obj, results_structure, model_name, split_name, varargin)
            % Plots grouped observed/predicted matrices from ratings_all.plot

            p = inputParser;
            addParameter(p, 'RunCondition', '', @(x) ischar(x) || isstring(x));
            parse(p, varargin{:});
            args = p.Results;

            split_result = obj.get_model_result(results_structure, split_name, model_name);
            ratings_all = obj.extract_ratings_all(split_result, split_name);

            available_conditions = fieldnames(ratings_all.plot);
            run_condition = obj.resolve_run_condition(args.RunCondition, available_conditions);

            y_mat = ratings_all.plot.(run_condition).y;
            yfit_mat = ratings_all.plot.(run_condition).yfit;

            figure;
            hold on;

            for i = 1:size(y_mat, 1)
                plot(y_mat(i, :), '-', 'LineWidth', 1);
                plot(yfit_mat(i, :), '--', 'LineWidth', 1);
            end

            hold off;
            xlabel('Bin');
            ylabel('Outcome');
            title(sprintf('%s | %s | %s grouped predictions', ...
                model_name, obj.pretty_split_name(split_name), run_condition), 'Interpreter', 'none');
            grid on;
        end

        function plot_model_weights(obj, results_structure, model_name, varargin)
            % Plots model weights for one model

            p = inputParser;
            addParameter(p, 'Source', 'testing', @(x) ischar(x) || isstring(x));
            addParameter(p, 'PC', [], @isnumeric);
            addParameter(p, 'MaxPoints', 5000, @isscalar);
            parse(p, varargin{:});
            args = p.Results;

            source_name = lower(char(string(args.Source)));

            switch source_name
                case 'testing'
                    testing_result = obj.get_model_result(results_structure, 'testing', model_name);

                    if ~isfield(testing_result, 'model_weights')
                        error('Testing result does not contain model_weights for model: %s', model_name);
                    end

                    weights = testing_result.model_weights;
                    plot_title = sprintf('%s | testing weights (best PC = %d)', ...
                        model_name, testing_result.best_PC);

                case 'training'
                    weights_table = obj.get_model_result(results_structure, 'model_weights', model_name);

                    if isempty(args.PC)
                        error('For Source = training, please provide a PC value.');
                    end

                    row_idx = weights_table.PC == args.PC;
                    if ~any(row_idx)
                        error('PC %d not found in model_weights table for model: %s', args.PC, model_name);
                    end

                    weights = weights_table.model_weights{row_idx};
                    plot_title = sprintf('%s | training weights (PC = %d)', model_name, args.PC);

                otherwise
                    error('Unknown weight source: %s', source_name);
            end

            weights = weights(:);

            if numel(weights) > args.MaxPoints
                idx = round(linspace(1, numel(weights), args.MaxPoints));
                weights_to_plot = weights(idx);
                xvals = idx;
            else
                weights_to_plot = weights;
                xvals = 1:numel(weights);
            end

            figure;
            plot(xvals, weights_to_plot, 'LineWidth', 1);
            xlabel('Feature Index');
            ylabel('Weight');
            title(plot_title, 'Interpreter', 'none');
            grid on;
        end

        function summary_table = summarize_all_models(~, results_structure, split_name)
            % Creates a compact performance summary table across all models

            split_name = lower(char(string(split_name)));

            if ~isfield(results_structure, split_name)
                error('Split "%s" not found in results_structure.', split_name);
            end

            model_names = fieldnames(results_structure.(split_name));
            summary_table = table();

            for i = 1:numel(model_names)
                model_name = model_names{i};
                result_obj = results_structure.(split_name).(model_name);

                switch split_name
                    case 'validation'
                        perf_table = result_obj.performance_table;
                        best_PC = result_obj.best_PC;
                        row_idx = perf_table.PC == best_PC;

                        if ~any(row_idx)
                            continue;
                        end

                        temp = perf_table(row_idx, {'mean_r', 'corr_y', 'rsquare', 'mse'});
                        temp.model_name = {model_name};
                        temp.best_PC = best_PC;
                        temp = movevars(temp, {'model_name', 'best_PC'}, 'Before', 1);

                    case 'testing'
                        temp = table( ...
                            {model_name}, result_obj.best_PC, result_obj.mean_r, ...
                            result_obj.corr_y, result_obj.rsquare, result_obj.mse, ...
                            'VariableNames', {'model_name', 'best_PC', 'mean_r', 'corr_y', 'rsquare', 'mse'});

                    otherwise
                        error('Summary is currently supported only for validation and testing.');
                end

                summary_table = [summary_table; temp];
            end
        end

        function save_summary_table(~, summary_table, save_path)
            % Saves summary table to .mat and .csv when possible

            save(save_path, 'summary_table', '-v7.3');

            [folder_path, file_stem, ~] = fileparts(save_path);
            csv_path = fullfile(folder_path, [file_stem '.csv']);

            try
                writetable(summary_table, csv_path);
            catch
                warning('Could not save CSV summary table: %s', csv_path);
            end
        end
    end

    methods (Access = private)

        function result_obj = get_model_result(~, results_structure, split_name, model_name)
            % Safely retrieves one model result block

            split_name = lower(char(string(split_name)));

            if ~isfield(results_structure, split_name)
                error('Split "%s" not found in results_structure.', split_name);
            end

            if ~isfield(results_structure.(split_name), model_name)
                error('Model "%s" not found in results_structure.%s.', model_name, split_name);
            end

            result_obj = results_structure.(split_name).(model_name);
        end

        function ratings_all = extract_ratings_all(~, split_result, split_name)
            % Extracts ratings_all depending on split structure

            split_name = lower(char(string(split_name)));

            switch split_name
                case 'training'
                    if ~isfield(split_result, 'ratings_all')
                        error('Training result does not contain ratings_all.');
                    end
                    ratings_all = split_result.ratings_all{1};

                case 'validation'
                    if ~isfield(split_result, 'performance_table') || isempty(split_result.performance_table)
                        error('Validation result does not contain performance_table.');
                    end

                    best_PC = split_result.best_PC;
                    row_idx = split_result.performance_table.PC == best_PC;

                    if ~any(row_idx)
                        error('Best PC %d not found in validation performance_table.', best_PC);
                    end

                    ratings_all = split_result.performance_table.ratings_all{row_idx};

                case 'testing'
                    if ~isfield(split_result, 'ratings_all')
                        error('Testing result does not contain ratings_all.');
                    end
                    ratings_all = split_result.ratings_all;

                otherwise
                    error('Unsupported split for ratings_all extraction: %s', split_name);
            end
        end

        function validate_metric(~, table_obj, metric_name, split_name, model_name)
            % Checks whether requested metric exists in table

            if ~ismember(metric_name, table_obj.Properties.VariableNames)
                error('Metric "%s" not found for %s | %s.', metric_name, split_name, model_name);
            end
        end

        function run_condition = resolve_run_condition(~, requested_condition, available_conditions)
            % Resolves which run condition to use for plotting

            if isempty(requested_condition)
                if numel(available_conditions) == 1
                    run_condition = available_conditions{1};
                else
                    error(['Multiple run conditions available. Please specify RunCondition explicitly. ' ...
                           'Available: %s'], strjoin(available_conditions, ', '));
                end
            else
                run_condition = char(string(requested_condition));

                if ~ismember(run_condition, available_conditions)
                    error('Requested RunCondition "%s" not available.', run_condition);
                end
            end
        end

        function split_label = pretty_split_name(~, split_name)
            % Prettifies split label for titles

            split_name = lower(char(string(split_name)));

            switch split_name
                case 'training'
                    split_label = 'Training';
                case 'validation'
                    split_label = 'Validation';
                case 'testing'
                    split_label = 'Testing';
                otherwise
                    split_label = split_name;
            end
        end
    end
end

classdef BehavioralProcessor < handle
    % BEHAVIORALPROCESSOR prepares behavioral targets for the modular fMRI prediction pipeline
    %
    % Goals:
    %   - Load continuous behavioral ratings
    %   - Interpolate ratings to TR resolution
    %   - Remove stimulus-delivery period if requested
    %   - Bin task and rest ratings using pipeline window settings
    %   - Save processed behavioral outputs aligned to fMRI bins
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   processor = BehavioralProcessor(config);
    %   processor.prepare_all();
    %
    % Inputs:
    %   - PipelineConfig object
    %   - Continuous behavioral rating files
    %
    % Outputs:
    %   - task_behavioral_ratings
    %   - rest_behavioral_ratings
    %
    % Author: Anel Zhunussova
    
    properties
        Config      % PipelineConfig object
    end

    methods
        function obj = BehavioralProcessor(config_obj)
            obj.Config = config_obj;
        end

        function processed_behavioral = prepare_all(obj, varargin)
            % Main behavioral preparation workflow for all scan units

            p = inputParser;
            addParameter(p, 'SaveResults', true, @islogical);
            parse(p, varargin{:});
            args = p.Results;

            processed_behavioral = struct();

            for i = 1:numel(obj.Config.Participants)
                scan_unit_id = obj.Config.Participants{i};

                if obj.Config.Verbose
                    fprintf('--- Preparing behavioral data: %s ---\n', scan_unit_id);
                end

                task_file = obj.get_behavioral_input_file(scan_unit_id, 'task');
                rest_file = obj.get_behavioral_input_file(scan_unit_id, 'rest');

                task_ratings = obj.load_continuous_ratings(task_file);
                rest_ratings = obj.load_continuous_ratings(rest_file);

                task_tr = obj.interpolate_to_tr(task_ratings);
                rest_tr = obj.interpolate_to_tr(rest_ratings);

                task_tr = obj.apply_task_timing_rules(task_tr);
                rest_tr = obj.apply_rest_timing_rules(rest_tr);

                task_behavioral_ratings = obj.bin_ratings(task_tr, obj.Config.RunConfig.task.WindowSizeTR, obj.Config.RunConfig.task.NumBins);
                rest_behavioral_ratings = obj.bin_ratings(rest_tr, obj.Config.RunConfig.rest.WindowSizeTR, obj.Config.RunConfig.rest.NumBins);

                safe_id = matlab.lang.makeValidName(scan_unit_id);

                processed_behavioral.(safe_id) = struct();
                processed_behavioral.(safe_id).scan_unit_id = scan_unit_id;
                processed_behavioral.(safe_id).task_behavioral_ratings = task_behavioral_ratings;
                processed_behavioral.(safe_id).rest_behavioral_ratings = rest_behavioral_ratings;

                if args.SaveResults
                    obj.save_processed_behavioral(scan_unit_id, task_behavioral_ratings, rest_behavioral_ratings);
                end
            end
        end
    end

    methods (Access = private)

        function input_file = get_behavioral_input_file(obj, scan_unit_id, run_type)
            % Returns expected raw behavioral input file path

            input_dir = fullfile(obj.Config.MetadataDir, 'raw_behavioral', scan_unit_id);

            switch lower(run_type)
                case 'task'
                    input_file = fullfile(input_dir, 'task_continuous_ratings.mat');
                case 'rest'
                    input_file = fullfile(input_dir, 'rest_continuous_ratings.mat');
                otherwise
                    error('Unknown run type: %s', run_type);
            end
        end

        function ratings = load_continuous_ratings(~, input_file)
            % Loads continuous ratings matrix with columns:
            % [time_in_seconds, rating_value]

            if ~exist(input_file, 'file')
                error('Behavioral input file not found: %s', input_file);
            end

            S = load(input_file);

            if isfield(S, 'continuous_rating')
                ratings = S.continuous_rating;
            elseif isfield(S, 'data') && isfield(S.data, 'dat') && isfield(S.data.dat, 'continuous_rating')
                ratings = S.data.dat.continuous_rating;
            else
                error('Could not find continuous_rating variable in file: %s', input_file);
            end

            if size(ratings, 2) < 2
                error('Continuous ratings must have at least two columns: time and rating.');
            end
        end

        function ratings_tr = interpolate_to_tr(obj, ratings)
            % Interpolates continuous ratings to TR-based samples

            time_difference = ratings(:, 1) - ratings(1, 1);
            num_TRs = ceil(max(time_difference) / obj.Config.TR);

            ratings_tr = zeros(num_TRs, 1);

            for t = 1:num_TRs
                time_start = obj.Config.TR * (t - 1);
                time_end = obj.Config.TR * t;
                idx = find(time_difference >= time_start & time_difference < time_end);

                if isempty(idx)
                    ratings_tr(t) = NaN;
                else
                    ratings_tr(t) = mean(ratings(idx, 2), 'omitnan');
                end
            end

            % Fill missing values linearly when possible
            valid_idx = ~isnan(ratings_tr);
            if sum(valid_idx) >= 2
                ratings_tr = interp1(find(valid_idx), ratings_tr(valid_idx), 1:num_TRs, 'linear', 'extrap')';
            end
        end

        function task_tr = apply_task_timing_rules(obj, task_tr)
            % Applies task-specific timing rules from config
            %
            % Public example:
            %   - task = capsaicin sustained-pain run
            %   - optional removal of stimulus-delivery period (for motion-related artifact removal)

            cfg = obj.Config.RunConfig.task;

            max_tr = min(numel(task_tr), cfg.MaxTR);
            task_tr = task_tr(1:max_tr);

            if isfield(cfg, 'RemoveStimulusDelivery') && cfg.RemoveStimulusDelivery
                stim_start_tr = floor(cfg.StimulusDeliveryStartSec / obj.Config.TR) + 1;
                stim_end_tr = floor(cfg.StimulusDeliveryEndSec / obj.Config.TR);

                keep_idx = true(numel(task_tr), 1);
                keep_idx(stim_start_tr:min(stim_end_tr, numel(task_tr))) = false;
                task_tr = task_tr(keep_idx);
            end
        end

        function rest_tr = apply_rest_timing_rules(obj, rest_tr)
            % Applies rest-specific timing rules from config

            cfg = obj.Config.RunConfig.rest;
            max_tr = min(numel(rest_tr), cfg.MaxTR);
            rest_tr = rest_tr(1:max_tr);
        end

        function binned_ratings = bin_ratings(~, ratings_tr, window_size_TR, num_bins)
            % Bins TR ratings into window-level behavioral targets

            max_possible_bins = floor(numel(ratings_tr) / window_size_TR);
            num_bins = min(num_bins, max_possible_bins);

            if num_bins < 1
                error('Not enough TRs to create even one behavioral bin.');
            end

            binned_ratings = zeros(num_bins, 1);

            for b = 1:num_bins
                tr_start = (b - 1) * window_size_TR + 1;
                tr_end = b * window_size_TR;
                binned_ratings(b) = mean(ratings_tr(tr_start:tr_end), 'omitnan');
            end
        end

        function save_processed_behavioral(obj, scan_unit_id, task_behavioral_ratings, rest_behavioral_ratings)
            % Saves processed behavioral outputs in the format expected by DataAssembler

            output_file = fullfile(obj.Config.MetadataDir, [scan_unit_id '_behavioral.mat']);

            save(output_file, ...
                'task_behavioral_ratings', ...
                'rest_behavioral_ratings', ...
                '-v7.3');
        end
    end
end

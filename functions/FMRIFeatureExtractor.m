classdef FMRIFeatureExtractor < handle
    % FMRIFEATUREEXTRACTOR extracts predictive features from preprocessed fMRI task and rest runs
    %
    % Goals:
    %   - Load preprocessed fMRI runs
    %   - Compute dynamic connectivity features using DCC
    %   - Extract HRF-based activation features
    %   - Save feature matrices for downstream modeling
    %
    % Public example:
    %   - task = capsaicin sustained-pain run
    %   - rest = resting-state run
    %
    % Sample usage:
    %   config = PipelineConfig();
    %   extractor = FMRIFeatureExtractor(config);
    %   extractor.extract_all();
    %
    % Inputs:
    %   - PipelineConfig object
    %   - Preprocessed fMRI time-series
    %
    % Outputs:
    %   - DCC connectivity features
    %   - HRF ROI activation features (intermediate output used for DCC calculation) 
    %   - HRF voxel activation features
    %
    % Author: Anel Zhunussova

    properties
        Config              % PipelineConfig object
        AtlasPath           % Path to brain atlas (.nii)
        GrayMatterMaskPath  % Path to gray matter mask (.nii)
    end

    methods
        function obj = FMRIFeatureExtractor(config_obj)
            % Initialize extractor with pipeline configuration.
            % Verifies that atlas and gray matter mask are accessible.

            obj.Config = config_obj;
            obj.validate_config();

            obj.AtlasPath = which([obj.Config.AtlasName '.nii']);
            if isempty(obj.AtlasPath)
                error(['Atlas file not found: ' obj.Config.AtlasName '.nii. ' ...
                       'Ensure atlas file is available on the MATLAB path.']);
            end

            obj.GrayMatterMaskPath = which(obj.Config.MaskName);
            if isempty(obj.GrayMatterMaskPath)
                error(['Gray matter mask not found: ' obj.Config.MaskName '. ' ...
                       'Ensure mask file is available on the MATLAB path.']);
            end
        end

        function extract_features(obj, varargin)
            % Main execution loop for feature extraction.
            % Iterates through participants, runs, and extraction methods.

            p = inputParser;
            addParameter(p, 'Methods', obj.Config.DefaultMethods, @iscell);
            addParameter(p, 'Runs', obj.Config.RunLabels, @iscell);
            addParameter(p, 'ApplyDurationCut', true, @islogical);
            addParameter(p, 'FolderSuffix', '', @(x) ischar(x) || isstring(x));
            parse(p, varargin{:});
            args = p.Results;

            for i = 1:numel(obj.Config.Participants)
                sub_id = obj.Config.Participants{i};

                if obj.Config.Verbose
                    fprintf('--- Processing Subject: %s ---\n', sub_id);
                end

                sub_save_dir = obj.setup_directories(sub_id, args.FolderSuffix);

                for r = 1:numel(args.Runs)
                    run_type = lower(args.Runs{r});
                    obj.process_subject_runs(sub_id, sub_save_dir, run_type, args);
                end
            end
        end
    end

    methods (Access = private)

        function validate_config(obj)
            % Checks that required configuration properties exist.

            required_props = { ...
                'RawDataDir', 'ResultsDir', 'Participants', 'RunLabels', ...
                'AtlasName', 'MaskName', 'DenoisingMethod', 'TR', ...
                'RunConfig', 'DefaultMethods'};

            for i = 1:numel(required_props)
                if ~isprop(obj.Config, required_props{i})
                    error('Missing required config property: %s', required_props{i});
                end
            end
        end

        function sub_save_dir = setup_directories(obj, sub_id, suffix)
            % Creates standardized output structure for Activation and DCC outputs.

            suffix = char(string(suffix));

            if isempty(strtrim(suffix))
                folder_name = sprintf('%s_%s', ...
                    obj.Config.DenoisingMethod, obj.Config.AtlasName);
            else
                folder_name = sprintf('%s_%s_%s', ...
                    obj.Config.DenoisingMethod, obj.Config.AtlasName, suffix);
            end

            sub_save_dir = fullfile(obj.Config.ResultsDir, folder_name, sub_id);

            activation_dir = fullfile(sub_save_dir, 'data', 'Activation');
            dcc_dir = fullfile(sub_save_dir, 'data', 'DCC');

            if ~exist(activation_dir, 'dir')
                mkdir(activation_dir);
            end

            if ~exist(dcc_dir, 'dir')
                mkdir(dcc_dir);
            end
        end

        function process_subject_runs(obj, sub_id, sub_save_dir, run_type, args)
            % Manages run discovery, image loading, denoising, and extraction.

            sub_func_dir = fullfile(obj.Config.RawDataDir, sub_id, 'func');

            if ~exist(sub_func_dir, 'dir')
                warning('Functional directory not found for subject: %s', sub_id);
                return;
            end

            run_pattern = obj.Config.get_run_pattern(run_type);
            files = dir(fullfile(sub_func_dir, run_pattern));

            if isempty(files)
                warning('No matching functional runs found for %s (%s).', sub_id, run_type);
                return;
            end

            for f = 1:numel(files)
                run_file = files(f).name;
                run_path = fullfile(files(f).folder, run_file);
                run_name = obj.strip_ext(run_file);

                if obj.Config.Verbose
                    fprintf('   Processing run: %s\n', run_name);
                end

                fmri_img = fmri_data(run_path, obj.GrayMatterMaskPath);
                fmri_img = obj.apply_denoising(fmri_img, sub_id, run_type, run_name);

                obj.run_extraction_algorithms(fmri_img, sub_save_dir, run_type, run_name, args);
            end
        end

        function fmri_img = apply_denoising(obj, fmri_img, sub_id, run_type, run_name)
            % Loads nuisance matrix corresponding to run and denoising strategy.

            nuis_dir = fullfile(obj.Config.RawDataDir, sub_id, 'nuisance_mat');

            nuis_candidates = { ...
                sprintf('nuisance_%s_%s_%s.mat', obj.Config.DenoisingMethod, run_type, run_name), ...
                sprintf('nuisance_%s_%s.mat', obj.Config.DenoisingMethod, run_type)};

            nuis_path = '';

            for i = 1:numel(nuis_candidates)
                candidate_path = fullfile(nuis_dir, nuis_candidates{i});
                if exist(candidate_path, 'file')
                    nuis_path = candidate_path;
                    break;
                end
            end

            if isempty(nuis_path)
                warning('Nuisance file not found for %s (%s). Proceeding without covariates.', ...
                    sub_id, run_type);
                return;
            end

            nuis_data = load(nuis_path);

            if ~isfield(nuis_data, 'R')
                error('Nuisance file does not contain variable R: %s', nuis_path);
            end

            if size(nuis_data.R, 1) ~= size(fmri_img.dat, 2)
                error(['Nuisance matrix row count does not match number of TRs. ' ...
                       'File: %s'], nuis_path);
            end

            fmri_img.covariates = nuis_data.R;
        end

        function run_extraction_algorithms(obj, fmri_img, sub_save_dir, run_type, run_name, args)
            % Integrated extraction logic for connectivity and HRF-based activation.

            for e = 1:numel(args.Methods)
                method = lower(args.Methods{e});

                switch method
                    case 'dcc'
                        obj.extract_dcc(fmri_img, sub_save_dir, run_name);

                    case 'hrf_roi'
                        obj.extract_hrf_betas(fmri_img, sub_save_dir, run_type, run_name, ...
                            args.ApplyDurationCut, 'roi');

                    case 'hrf_voxel'
                        obj.extract_hrf_betas(fmri_img, sub_save_dir, run_type, run_name, ...
                            args.ApplyDurationCut, 'voxel');

                    otherwise
                        error('Unknown extraction method: %s', method);
                end
            end
        end

        function extract_dcc(obj, fmri_img, sub_save_dir, run_name)
            % Performs ROI extraction internally and computes Dynamic Conditional Correlation.
            % Only DCC output is saved in the public pipeline.

            [~, roi_obj] = canlab_connectivity_preproc(fmri_img, ...
                'windsorize', 5, ...
                'lpf', 0.1, obj.Config.TR, ...
                'extract_roi', obj.AtlasPath, ...
                'no_plots');

            roi_data = roi_obj{1}.dat;
            dcc_data = DCC_jj(roi_data, 'simple', 'whiten');

            save(fullfile(sub_save_dir, 'data', 'DCC', ...
                ['dcc_' run_name '.mat']), 'dcc_data', '-v7.3');
        end

        function extract_hrf_betas(obj, fmri_img, sub_save_dir, run_type, run_name, cut_duration, beta_mode)
            % Performs single-trial HRF regression for ROI or voxel outputs.

            total_TR = size(fmri_img.dat, 2);
            onsets = obj.calculate_onsets(total_TR, run_type, cut_duration);

            if isempty(onsets)
                warning('No valid onsets generated for run: %s', run_name);
                return;
            end

            X = plotDesign_mint(onsets, [], obj.Config.TR, total_TR * obj.Config.TR, ...
                'samefig', spm_hrf(obj.Config.TR), 'singletrial');

            num_bins = numel(onsets);

            % Preallocate output matrix using first regression pass
            if strcmpi(beta_mode, 'roi')
                [~, ~, ~, ~, r_beta] = canlab_connectivity_preproc(fmri_img, ...
                    'windsorize', 5, ...
                    'lpf', 0.1, obj.Config.TR, ...
                    'extract_roi', obj.AtlasPath, ...
                    'no_plots', ...
                    'regressors', X(:, 1));

                beta_results = zeros(size(r_beta{1}.dat, 1), num_bins);

            elseif strcmpi(beta_mode, 'voxel')
                [~, ~, ~, v_beta, ~] = canlab_connectivity_preproc(fmri_img, ...
                    'windsorize', 5, ...
                    'lpf', 0.1, obj.Config.TR, ...
                    'extract_roi', obj.AtlasPath, ...
                    'no_plots', ...
                    'regressors', X(:, 1));

                beta_results = zeros(size(v_beta.dat, 1), num_bins);

            else
                error('Unknown beta mode: %s', beta_mode);
            end

            for b = 1:num_bins
                [~, ~, ~, v_beta, r_beta] = canlab_connectivity_preproc(fmri_img, ...
                    'windsorize', 5, ...
                    'lpf', 0.1, obj.Config.TR, ...
                    'extract_roi', obj.AtlasPath, ...
                    'no_plots', ...
                    'regressors', X(:, b));

                if strcmpi(beta_mode, 'roi')
                    beta_results(:, b) = r_beta{1}.dat;
                else
                    beta_results(:, b) = v_beta.dat;
                end
            end

            save(fullfile(sub_save_dir, 'data', 'Activation', ...
                ['hrf_' lower(beta_mode) '_' run_name '.mat']), ...
                'beta_results', '-v7.3');
        end

        function onsets = calculate_onsets(obj, total_TR, run_type, cut_duration)
            % Generates temporal windows based on run-specific configuration.
            %
            % The public example assumes:
            %   - task = capsaicin sustained-pain run
            %   - rest = resting-state run
            %
            % When cut_duration is true for task runs, the configured stimulus
            % delivery interval is removed before generating bins.

            run_cfg = obj.Config.get_run_config(run_type);

            start_tr = run_cfg.StartTR;
            end_tr = min(total_TR, run_cfg.MaxTR);
            window_size = run_cfg.WindowSizeTR;
            requested_bins = run_cfg.NumBins;

            if end_tr < start_tr
                warning('Invalid run range for run type: %s', run_type);
                onsets = {};
                return;
            end

            if strcmpi(run_type, 'task')
                if isfield(run_cfg, 'RemoveStimulusDelivery') && ...
                        run_cfg.RemoveStimulusDelivery && cut_duration

                    stim_end_tr = floor(run_cfg.StimulusDeliveryEndSec / obj.Config.TR);
                    start_tr = max(start_tr, stim_end_tr + 1);
                end
            end

            available_TRs = end_tr - start_tr + 1;
            max_possible_bins = floor(available_TRs / window_size);
            num_bins = min(requested_bins, max_possible_bins);

            if num_bins < 1
                warning('No complete bins available for run type: %s', run_type);
                onsets = {};
                return;
            end

            start_trs = start_tr + (0:(num_bins - 1)) * window_size;
            onsets = cell(num_bins, 1);

            for b = 1:num_bins
                onsets{b} = [(start_trs(b) - 1) * obj.Config.TR, ...
                             window_size * obj.Config.TR];
            end
        end

        function stem = strip_ext(~, filename)
            % Removes .nii or .nii.gz extension from filename.

            stem = filename;

            if endsWith(stem, '.nii.gz')
                stem = erase(stem, '.nii.gz');
            elseif endsWith(stem, '.nii')
                stem = erase(stem, '.nii');
            end
        end
    end
end

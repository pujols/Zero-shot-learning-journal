function EXEM_ImageNet_comb(task, opt, direct_test, feature_name, semantic_name)

%% Input
% task: 'train', 'test'
% opt: opt.C: the regularizer coefficient of nu-SVR (e.g, 2 .^ 0)
%      opt.nu: the nu-SVR parameter (e.g, 2 .^ (-10 : 0))
%      opt.gamma: the RBF scale parameter (e.g., 2 .^ (-5 : 5))
%      opt.pca_d: the PCA dimensionality (e.g., 500)
%      opt.comb_weight: in [0, 1]
%      opt.comb_scale
%      opt.GZSL: '_GZSL' or []
%      opt.downsample
%      opt.nr_fold
%      opt.test_type: 'dis', 'eu', 'seu'
%      opt.EXEM = []
% direct_test: test on a specific [C, nu, gamma, pca_d] pair without cross-validation

%% Settings
if ~isfield(opt, 'downsample')
    opt.downsample = 1;
end
    
set_path_EXEM;
norm_method = 'L2';
if isfield(opt, 'nr_fold')
    nr_fold = opt.nr_fold;
else
    nr_fold = 5;
end

%% Data
[Xtr, Ytr, Xhold_all, Yhold_all, attr2, class_order, CV_hold_ind, CV_hold_loc] = data_loader_ImageNet1K(feature_name, semantic_name, opt);
Sig_Y = get_class_signatures(attr2, norm_method);

attr2_MDS = attribute_loader_ImageNet1K(feature_name, 'MDS', opt);
Sig_Y_MDS = get_class_signatures(attr2_MDS, norm_method);

if ~isempty(semantic_name)
    semantic_name = ['_', semantic_name];
end


%% 5-fold class-wise cross validation splitting (for 'train' and 'val')
[fold_loc, hold_fold_loc] = cv_split_ImageNet1K(task, Ytr, class_order, CV_hold_ind, CV_hold_loc);

%% training and validation
if strcmp(task, 'train')
    acc_val_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    auc_val_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    HM_val_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    fixed_bias_val_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    acc_val_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    auc_val_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    HM_val_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    fixed_bias_val_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
    dis_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d), length(opt.comb_scale), length(opt.comb_weight));
        
    for f = 1 : nr_fold
        Xbase = Xtr;
        Xbase(fold_loc{f}, :) = [];
        Ybase = Ytr;
        Ybase(fold_loc{f}) = [];
        Xval = Xtr(fold_loc{f}, :);
        Yval = Ytr(fold_loc{f});
        Xhold = Xhold_all;
        Xhold(hold_fold_loc{f}, :) = [];
        Yhold = Yhold_all;
        Yhold(hold_fold_loc{f}) = [];
        
        Xval = Xval(1 : opt.downsample : end, :);
        Yval = Yval(1 : opt.downsample : end);
        Xhold = Xhold(1 : opt.downsample : end, :);
        Yhold = Yhold(1 : opt.downsample : end);
        
        % Xcomb = [Xhold; Xval];
        % Ycomb = [Yhold; Yval];
        label_S = unique(Yhold);
        label_U = unique(Yval);
        Sig_Ybase = Sig_Y(unique(Ybase), :);
        Sig_Yhold = Sig_Y(unique(Yhold), :);
        Sig_Yval = Sig_Y(unique(Yval), :);
        Sig_Ybase_MDS = Sig_Y_MDS(unique(Ybase), :);
        Sig_Yhold_MDS = Sig_Y_MDS(unique(Yhold), :);
        Sig_Yval_MDS = Sig_Y_MDS(unique(Yval), :);
        
        %% Starting of EXEM training
        % PCA learning
        [mean_Xbase_PCA, V] = do_pca(Xbase);
        Xbase = bsxfun(@minus, Xbase, mean_Xbase_PCA);
        Xhold = bsxfun(@minus, Xhold, mean_Xbase_PCA);
        Xval = bsxfun(@minus, Xval, mean_Xbase_PCA);
        % PCA projection
        Xbase = Xbase * V;
        Xhold = Xhold * V;
        Xval = Xval * V;
        
        pdist_bb = pdist2_fast(Sig_Ybase, Sig_Ybase);
        pdist_hb = pdist2_fast(Sig_Yhold, Sig_Ybase);
        pdist_vb = pdist2_fast(Sig_Yval, Sig_Ybase);
        pdist_bb_MDS = pdist2_fast(Sig_Ybase_MDS, Sig_Ybase_MDS);
        pdist_hb_MDS = pdist2_fast(Sig_Yhold_MDS, Sig_Ybase_MDS);
        pdist_vb_MDS = pdist2_fast(Sig_Yval_MDS, Sig_Ybase_MDS);
        
        for g = 1: length(opt.gamma)
            % SVR kernel
            for cs = 1 : length(opt.comb_scale)
                for cw = 1 : length(opt.comb_weight)
                    Ker_base = [(1 : length(unique(Ybase)))', opt.comb_weight(cw) * exp(-opt.gamma(g) * pdist_bb .^ 2)...
                        + (1 - opt.comb_weight(cw)) * exp(-opt.comb_scale(cs) * pdist_bb_MDS .^ 2)];
                    Ker_hold = [(1 : length(unique(Yhold)))', opt.comb_weight(cw) * exp(-opt.gamma(g) * pdist_hb .^ 2)...
                        + (1 - opt.comb_weight(cw)) * exp(-opt.comb_scale(cs) * pdist_hb_MDS .^ 2)];
                    Ker_val = [(1 : length(unique(Yval)))', opt.comb_weight(cw) * exp(-opt.gamma(g) * pdist_vb .^ 2)...
                        + (1 - opt.comb_weight(cw)) * exp(-opt.comb_scale(cs) * pdist_vb_MDS .^ 2)];
        
                    for d = 1 : length(opt.pca_d)
                        disp([f, g, d])
                        [mean_Xbase, std_Xbase] = compute_class_stat(Ybase, Xbase(:, 1 : opt.pca_d(d)));
                        avg_std_Xbase = mean(std_Xbase, 1);

                        for c = 1 : length(opt.C)
                            for n = 1 : length(opt.nu)
                                cur_C = opt.C(c); cur_nu = opt.nu(n);
                                mean_Xbase_rec = zeros(size(Sig_Ybase, 1), opt.pca_d(d));
                                mean_Xhold_rec = zeros(size(Sig_Yhold, 1), opt.pca_d(d));
                                mean_Xhold = zeros(size(Sig_Yhold, 1), opt.pca_d(d));
                                mean_Xval_rec = zeros(size(Sig_Yval, 1), opt.pca_d(d));
                                mean_Xval = zeros(size(Sig_Yval, 1), opt.pca_d(d));

                                parfor j = 1 : opt.pca_d(d)
                                    % SVR learning and testing
                                    regressor = svmtrain(mean_Xbase(:, j), Ker_base, ['-s 4 -t 4 -c ' num2str(cur_C) ' -n ' num2str(cur_nu) ' -m 10000']);
                                    mean_Xbase_rec(:, j) = svmpredict(mean_Xbase(:, j), Ker_base, regressor);
                                    mean_Xhold_rec(:, j) = svmpredict(mean_Xhold(:, j), Ker_hold, regressor);
                                    mean_Xval_rec(:, j) = svmpredict(mean_Xval(:, j), Ker_val, regressor);
                                end

                                % Get scores
                                dis_eu(c, n, g, d, cs, cw) = test_EXEM_ImageNet1K(Xval(:, 1 : opt.pca_d(d)), Yval, mean_Xval_rec);
                                [score_S_eu, score_U_eu, score_S_seu, score_U_seu] =...
                                    test_EXEM_GZSL_ImageNet1K([Xhold(:, 1 : opt.pca_d(d)); Xval(:, 1 : opt.pca_d(d))], mean_Xhold_rec, mean_Xval_rec, avg_std_Xbase);

                                %% Euclidean
                                % ZSL performance
                                disp('ZSL evaluation!')
                                if strcmp(feature_name, 'googlenet') % sample-wise
                                    [~, ~, acc2] = evaluate_easy_ImageNet1K(score_U_eu(length(Yhold) + 1 : end, :), Yval, 1);
                                    acc_val_eu(c, n, g, d, cs, cw) = acc_val_eu(c, n, g, d, cs, cw) + acc2 / nr_fold;
                                elseif strcmp(feature_name, 'resnet') % class-wise         
                                    acc_val_eu(c, n, g, d, cs, cw) = acc_val_eu(c, n, g, d, cs, cw) + evaluate_easy_ImageNet1K(score_U_eu(length(Yhold) + 1 : end, :), Yval, 1) / nr_fold;
                                else
                                    disp('Wrong features!')
                                    return;
                                end

                                % GZSL performance
                                disp('GZSL evaluation!')
                                if strcmp(feature_name, 'googlenet') % sample-wise
                                    [auc, ~, ~, HM, fixed_bias] =...
                                        Compute_AUSUC_ImageNet1K(score_S_eu, score_U_eu, [Yhold; Yval], label_S, label_U, length(Yhold), length(Yval), 1, 'sample-wise', []);                
                                elseif strcmp(feature_name, 'resnet') % class-wise
                                    [auc, ~, ~, HM, fixed_bias] =...
                                        Compute_AUSUC_ImageNet1K(score_S_eu, score_U_eu, [Yhold; Yval], label_S, label_U, length(Yhold), length(Yval), 1, 'class-wise', []);
                                else
                                    disp('Wrong features!')
                                    return;
                                end

                                auc_val_eu(c, n, g, d, cs, cw) = auc_val_eu(c, n, g, d, cs, cw) + auc / nr_fold;
                                HM_val_eu(c, n, g, d, cs, cw) = HM_val_eu(c, n, g, d, cs, cw) + HM / nr_fold;
                                fixed_bias_val_eu(c, n, g, d, cs, cw) = fixed_bias_val_eu(c, n, g, d, cs, cw) + fixed_bias / nr_fold;    

                                %% standard Euclidean
                                % ZSL performance
                                disp('ZSL evaluation!')
                                if strcmp(feature_name, 'googlenet') % sample-wise
                                    [~, ~, acc2] = evaluate_easy_ImageNet1K(score_U_seu(length(Yhold) + 1 : end, :), Yval, 1);
                                    acc_val_seu(c, n, g, d, cs, cw) = acc_val_seu(c, n, g, d, cs, cw) + acc2 / nr_fold;
                                elseif strcmp(feature_name, 'resnet') % class-wise         
                                    acc_val_seu(c, n, g, d, cs, cw) = acc_val_seu(c, n, g, d, cs, cw) + evaluate_easy_ImageNet1K(score_U_seu(length(Yhold) + 1 : end, :), Yval, 1) / nr_fold;
                                else
                                    disp('Wrong features!')
                                    return;
                                end

                                % GZSL performance
                                disp('GZSL evaluation!')
                                if strcmp(feature_name, 'googlenet') % sample-wise
                                    [auc, ~, ~, HM, fixed_bias] =...
                                        Compute_AUSUC_ImageNet1K(score_S_seu, score_U_seu, [Yhold; Yval], label_S, label_U, length(Yhold), length(Yval), 1, 'sample-wise', []);                
                                elseif strcmp(feature_name, 'resnet') % class-wise
                                    [auc, ~, ~, HM, fixed_bias] =...
                                        Compute_AUSUC_ImageNet1K(score_S_seu, score_U_seu, [Yhold; Yval], label_S, label_U, length(Yhold), length(Yval), 1, 'class-wise', []);
                                else
                                    disp('Wrong features!')
                                    return;
                                end

                                auc_val_seu(c, n, g, d, cs, cw) = auc_val_seu(c, n, g, d, cs, cw) + auc / nr_fold;
                                HM_val_seu(c, n, g, d, cs, cw) = HM_val_seu(c, n, g, d, cs, cw) + HM / nr_fold;
                                fixed_bias_val_seu(c, n, g, d, cs, cw) = fixed_bias_val_seu(c, n, g, d, cs, cw) + fixed_bias / nr_fold; 
                            end
                        end
                    end
                end
            end
        end
    end       

    save(['../EXEM_CV_results/EXEM_comb_classCV_ImageNet_' feature_name semantic_name '_' norm_method '.mat'],...
        'acc_val_eu' , 'auc_val_eu', 'HM_val_eu', 'fixed_bias_val_eu', 'acc_val_seu' , 'auc_val_seu', 'HM_val_seu', 'fixed_bias_val_seu', 'dis_eu', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    fixed_bias = 0;
    
    if isempty(direct_test)
        cur_GZSL = opt.GZSL;
        cur_test_type = opt.test_type;
        load(['../EXEM_CV_results/EXEM_comb_classCV_ImageNet_' feature_name semantic_name '_' norm_method '.mat'],...
        'acc_val_eu' , 'auc_val_eu', 'fixed_bias_val_eu', 'acc_val_seu' , 'auc_val_seu', 'fixed_bias_val_seu', 'dis_eu', 'opt');
        opt.GZSL = cur_GZSL;
        opt.test_type = cur_test_type;
        
        if strcmp(opt.test_type, 'dis')
            [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(-dis_eu);
        elseif isempty(opt.GZSL)
            if strcmp(opt.test_type, 'eu')
                [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(acc_val_eu);
            elseif strcmp(opt.test_type, 'seu')
                [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(acc_val_seu);
            else
                disp('Wrong test_type!');
                return;
            end
        else
            if strcmp(opt.test_type, 'eu')
                [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(auc_val_eu);
                fixed_bias = fixed_bias_val_eu(loc_C(1), loc_nu(1), loc_gamma(1), loc_pca_d(1), loc_cs(1), loc_cw(1));
            elseif strcmp(opt.test_type, 'seu')
                [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(auc_val_seu);
                fixed_bias = fixed_bias_val_seu(loc_C(1), loc_nu(1), loc_gamma(1), loc_pca_d(1), loc_cs(1), loc_cw(1));
            else
                disp('Wrong test_type!');
                return;
            end
            disp(fixed_bias);
        end
        
        C = opt.C(loc_C(1)); nu = opt.nu(loc_nu(1)); gamma = opt.gamma(loc_gamma(1)); pca_d = opt.pca_d(loc_pca_d(1));
        comb_scale = opt.comb_scale(loc_cs(1)); comb_weight = opt.comb_weight(loc_cw(1));
        disp([loc_C(1), loc_nu(1), loc_gamma(1), loc_pca_d(1), loc_cs(1), loc_cw(1), fixed_bias]);
    else
        
        C = direct_test(1); nu = direct_test(2); gamma = direct_test(3); pca_d = direct_test(4); comb_scale = direct_test(5); comb_weight = direct_test(6);
        if ~isempty(opt.GZSL)
            fixed_bias = direct_test(7);
        end
    end
    
    Xtr = [Xtr; Xhold_all];
	Ytr = [Ytr; Yhold_all];
    label_R = unique(Ytr);
    Sig_R = Sig_Y(label_R, :);
    Sig_R_MDS = Sig_Y_MDS(label_R, :);
    
    [mean_Xtr_PCA, V] = do_pca(Xtr);
    Xtr = bsxfun(@minus, Xtr, mean_Xtr_PCA);
    Xtr = Xtr * V(:, 1 : pca_d);
    [mean_Xtr, std_Xtr] = compute_class_stat(Ytr, Xtr);
    avg_std_Xtr = mean(std_Xtr, 1);
    
    Ker_S = [(1 : length(label_R))', comb_weight * exp(-gamma * pdist2_fast(Sig_R, Sig_R) .^ 2) + (1 - comb_weight) * exp(-comb_scale * pdist2_fast(Sig_R_MDS, Sig_R_MDS) .^ 2)];
    regressors = cell(1, pca_d);
    
    if strcmp(opt.test_type, 'dis')
        load /projects/vision/zero-shot-learning/data/ImageNet_w2v_skip_1.mat no_w2v_loc wnids words
        [~, Sig_S, Sig_U, ~, label_S, label_U] = attribute_loader_ImageNet(feature_name, semantic_name(2 : end), 'all', opt);
        [Sig_S, median_Dist] = get_class_signatures(Sig_S, norm_method);
        Sig_U = get_class_signatures_ImageNet_U(Sig_U, norm_method, median_Dist);
        [~, Sig_S_MDS, Sig_U_MDS, ~, ~, ~] = attribute_loader_ImageNet(feature_name, 'MDS', 'all', opt);
        [Sig_S_MDS, median_Dist] = get_class_signatures(Sig_S_MDS, norm_method);
        Sig_U_MDS = get_class_signatures_ImageNet_U(Sig_U_MDS, norm_method, median_Dist);
        Sig_Y = [Sig_S; Sig_U];
        Sig_Y_MDS = [Sig_S_MDS; Sig_U_MDS];
        Ker = [(1 : length(label_S) + length(label_U))',...
            comb_weight * exp(-gamma * pdist2_fast(Sig_Y, Sig_R) .^ 2) + (1 - comb_weight) * exp(-comb_scale * pdist2_fast(Sig_Y_MDS, Sig_R_MDS) .^ 2)];
        w2v = zeros(size(Sig_Y, 1), pca_d);
    end
    
    parfor j = 1 : pca_d
        % SVR learning and testing
        regressors{j} = svmtrain(mean_Xtr(:, j), Ker_S, ['-s 4 -t 4 -c ' num2str(C) ' -n ' num2str(nu) ' -m 10000']);
    end
    
    if strcmp(opt.test_type, 'dis')
        parfor j = 1 : pca_d
            w2v(:, j) = svmpredict(w2v(:, j), Ker, regressors{j});
        end
    end

    if strcmp(opt.test_type, 'dis')
        save (['/projects/vision/zero-shot-learning/data/ImageNet' semantic_name '_comb_' feature_name '_EXEM.mat'], 'w2v', 'no_w2v_loc', 'wnids', 'words', '-v7.3');
    else
        save(['../EXEM_ImageNet_classifiers/EXEM_comb' opt.GZSL '_' opt.test_type '_ImageNet_' feature_name semantic_name '_' norm_method '_C' num2str(C) '_nu' num2str(nu)...
            '_gamma' num2str(gamma) '_pca_d' num2str(pca_d) '.mat'],...
            'mean_Xtr_PCA' , 'V', 'regressors', 'pca_d', 'avg_std_Xtr', 'gamma', 'fixed_bias', 'comb_weight', 'comb_scale', '-v7.3');
    end
end

end

function [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = find_max(acc_mat)
    [~, max_ind] = max(acc_mat(:));
    [loc_C, loc_nu, loc_gamma, loc_pca_d, loc_cs, loc_cw] = ind2sub(size(acc_mat), max_ind);
end
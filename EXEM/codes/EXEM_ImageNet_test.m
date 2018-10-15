function EXEM_ImageNet_test(task, opt, direct_test, feature_name, semantic_name, flat_hit)

%% Input
% task: '2-hop', '3-hop', '4-hop', 'all', 'M-500', 'M-1K', 'M-5K', 'L-500',
% 'L-1K', 'L-5K'
% opt: opt.C: the regularizer coefficient of nu-SVR (e.g, 2 .^ 0)
%      opt.nu: the nu-SVR parameter (e.g, 2 .^ (-10 : 0))
%      opt.gamma: the RBF scale parameter (e.g., 2 .^ (-5 : 5))
%      opt.pca_d: the PCA dimensionality (e.g., 500)
%      opt.GZSL: '_GZSL' or []
%      opt.nr_fold
%      opt.test_type: 'dis', 'eu', 'seu'
%      opt.EXEM = []
%      opt.split
% direct_test: test on a specific [C, nu, gamma, pca_d] pair without cross-validation

%% Setting
set_path_EXEM;
norm_method = 'L2';

if isfield(opt, 'split')
    split_or_not = opt.split;
else
    split_or_not = [];
end

%% Load semantic embeddings
[Sig_R, Sig_S, Sig_U, labels_R, labels_S, labels_U] = attribute_loader_ImageNet(feature_name, semantic_name, task, opt);
[Sig_S, median_Dist] = get_class_signatures(Sig_S, norm_method);
Sig_R = get_class_signatures_ImageNet_U(Sig_R, norm_method, median_Dist);
Sig_U = get_class_signatures_ImageNet_U(Sig_U, norm_method, median_Dist);

if ~isempty(semantic_name)
    semantic_name = ['_', semantic_name];
end

%% Load model
if isempty(direct_test)
    cur_GZSL = opt.GZSL;
    cur_test_type = opt.test_type;
    load(['../EXEM_CV_results/EXEM_classCV_ImageNet_' feature_name semantic_name '_' norm_method '.mat'],...
    'acc_val_eu' , 'auc_val_eu', 'acc_val_seu' , 'auc_val_seu', 'opt');
    opt.GZSL = cur_GZSL;
    opt.test_type = cur_test_type;

    if isempty(opt.GZSL)
        if strcmp(opt.test_type, 'eu')
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(acc_val_eu);
        elseif strcmp(opt.test_type, 'seu')
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(acc_val_seu);
        else
            disp('Wrong test_type!');
            return;
        end
    else
        if strcmp(opt.test_type, 'eu')
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(auc_val_eu);
        elseif strcmp(opt.test_type, 'seu')
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(auc_val_seu);
        else
            disp('Wrong test_type!');
            return;
        end
    end
    clear acc_val_eu; clear auc_val_eu; clear acc_val_seu; clear auc_val_seu;
    C = opt.C(loc_C(1)); nu = opt.nu(loc_nu(1)); gamma = opt.gamma(loc_gamma(1)); pca_d = opt.pca_d(loc_pca_d(1));
else

    C = direct_test(1); nu = direct_test(2); gamma = direct_test(3); pca_d = direct_test(4);
end

load(['../EXEM_ImageNet_classifiers/EXEM' opt.GZSL '_' opt.test_type '_ImageNet_' feature_name semantic_name '_' norm_method '_C' num2str(C) '_nu' num2str(nu)...
            '_gamma' num2str(gamma) '_pca_d' num2str(pca_d) '.mat'], 'mean_Xtr_PCA' , 'V', 'regressors', 'avg_std_Xtr', 'fixed_bias');
    
if ~isempty(opt.GZSL) && ~isempty(direct_test)
    fixed_bias = direct_test(5);
end

%% Build up the model
Ker_S = [(1 : length(labels_S))', exp(-gamma * pdist2_fast(Sig_S, Sig_R) .^ 2)]; %%%
Ker_U = [(1 : length(labels_U))', exp(-gamma * pdist2_fast(Sig_U, Sig_R) .^ 2)]; %%%
mean_S_rec = zeros(length(labels_S), pca_d);
mean_U_rec = zeros(length(labels_U), pca_d);
parfor j = 1 : pca_d
    mean_S_rec(:, j) = svmpredict(mean_S_rec(:, j), Ker_S, regressors{j});
    mean_U_rec(:, j) = svmpredict(mean_U_rec(:, j), Ker_U, regressors{j});
end

%% Load data
[datafrom, loc_index] = data_loader_index_ImageNet(task, opt); %%%

for i = 1 : length(datafrom)
    disp(['at iteration ' num2str(i) ' of ' num2str(length(datafrom))]);
    disp('prediction');
    [Xte, Yte] = data_loader_ImageNet(feature_name, opt, labels_S, labels_U, datafrom(i), loc_index(i), split_or_not);
    Xte = bsxfun(@minus, Xte, mean_Xtr_PCA);
    Xte = Xte * V(:, 1 : pca_d);
    
    if ~isempty(opt.GZSL)
        if strcmp(opt.test_type, 'eu')
            score_S = -pdist2_fast(Xte, mean_S_rec);
        else
            score_S = -pdist2_fast(bsxfun(@rdivide, Xte, avg_std_Xtr), bsxfun(@rdivide, mean_S_rec, avg_std_Xtr));
        end
    end
    if strcmp(opt.test_type, 'eu')
        score_U = -pdist2_fast(Xte, mean_U_rec);
    else
        score_U = -pdist2_fast(bsxfun(@rdivide, Xte, avg_std_Xtr), bsxfun(@rdivide, mean_U_rec, avg_std_Xtr));
    end
    
    %% for efficiency without copying the data
    disp('evaluation');
    if isempty(opt.GZSL)
        local_labels = unique(Yte);
        pred_correct = zeros(1, length(flat_hit));
        acc_class = zeros(length(local_labels), length(flat_hit));

        if length(flat_hit) == 1 && flat_hit(1) == 1
            [~, Ypred_loc] = max(score_U, [], 2);
            temp_record = double(Yte == labels_U(Ypred_loc));
            pred_correct = sum(temp_record);
            for k = 1 : length(local_labels)
                acc_class(k) = sum((Yte == local_labels(k)) & (temp_record == 1)) / sum(Yte == local_labels(k));
            end

        else
            flat_hit_record = zeros(length(Yte), max(flat_hit));
            [~, Ypred_loc] = sort(score_U, 2, 'descend');
            for t = 1 : max(flat_hit)
                flat_hit_record(:, t) = double(Yte == labels_U(Ypred_loc(:, t)));
            end
            for j = 1 : length(flat_hit)
                temp_record = double(sum(flat_hit_record(:, 1 : flat_hit(j)), 2) > 0.5);   
                pred_correct(j) = sum(temp_record);
                for k = 1 : length(local_labels)
                    acc_class(k, j) = sum((Yte == local_labels(k)) & (temp_record == 1)) / sum(Yte == local_labels(k));
                end
            end
        end

        Y_size = length(Yte);
        save(['../EXEM_ImageNet_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method ...
            '_from' num2str(datafrom(i)) '_loc' num2str(loc_index(i)) '.mat'], 'pred_correct', 'acc_class', 'Y_size', 'flat_hit');
        
    else
        for p = 1 : length(flat_hit)
            [temp_bias, temp_Y_correct_S, temp_Y_correct_U, temp_class_correct_S, temp_class_count_S, temp_class_count_U, temp_loc_S, temp_loc_U]...
                = Compute_AUSUC_ImageNet_split(score_S, score_U, Yte, labels_S, labels_U, flat_hit(p));
            
            temp_N_S = 0; temp_N_U = 0;
            if datafrom(i) == 0
                temp_N_S = length(Yte);
            else
                temp_N_U = length(Yte);
            end
            
            save(['../EXEM_ImageNet_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method ...
                '_from' num2str(datafrom(i)) '_loc' num2str(loc_index(i)) '_flat' num2str(flat_hit(p)) '.mat'], 'temp_bias', 'temp_Y_correct_S', 'temp_Y_correct_U',...
                'temp_class_correct_S', 'temp_class_count_S', 'temp_class_count_U', 'temp_loc_S', 'temp_loc_U', 'temp_N_S', 'temp_N_U');
        end
    end
end

disp('merging');
if isempty(opt.GZSL)
    total_pred_correct = 0;
    total_acc_class = [];
    total_Y_size = 0;
    for i = 1 : length(datafrom)
        load(['../EXEM_ImageNet_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method ...
            '_from' num2str(datafrom(i)) '_loc' num2str(loc_index(i)) '.mat'], 'pred_correct', 'acc_class', 'Y_size', 'flat_hit');
            total_pred_correct = total_pred_correct + pred_correct;
            total_Y_size = total_Y_size + Y_size;
            total_acc_class = [total_acc_class; acc_class];        
    end
    acc_sample = total_pred_correct / total_Y_size;
    acc_class = mean(total_acc_class(~isnan(sum(total_acc_class, 2)), :), 1);
    save(['../EXEM_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method '.mat'],...
        'acc_class', 'acc_sample');
    disp([acc_class, acc_sample]);
    
else
    for p = 1 : length(flat_hit)
        bias = [];
        Y_correct_S = [];
        Y_correct_U = [];
        class_correct_S = 0;
        class_count_S = 0;
        class_count_U = 0;
        loc_S = [];
        loc_U = [];
        N_S = 0;
        N_U = 0;
        
        for i = 1 : length(datafrom)
            load(['../EXEM_ImageNet_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method ...
                    '_from' num2str(datafrom(i)) '_loc' num2str(loc_index(i)) '_flat' num2str(flat_hit(p)) '.mat'], 'temp_bias', 'temp_Y_correct_S', 'temp_Y_correct_U',...
                    'temp_class_correct_S', 'temp_class_count_S', 'temp_class_count_U', 'temp_loc_S', 'temp_loc_U', 'temp_N_S', 'temp_N_U');
            bias = [bias; temp_bias];
            Y_correct_S = [Y_correct_S;  temp_Y_correct_S];
            Y_correct_U = [Y_correct_U;  temp_Y_correct_U];
            loc_S = [loc_S; temp_loc_S];
            loc_U = [loc_U; temp_loc_U];
            class_correct_S = class_correct_S + temp_class_correct_S;
            class_count_S = class_count_S + temp_class_count_S;
            class_count_U = class_count_U + temp_class_count_U;
            N_S = N_S + temp_N_S;
            N_U = N_U + temp_N_U;
        end
        
        [AUC_val_class_wise, AUC_record_class_wise, acc_noBias_class_wise, acc_class_wise, HM_class_wise, AUC_val_sample_wise, AUC_record_sample_wise, acc_noBias_sample_wise,...
    acc_sample_wise, HM_sample_wise] = Compute_AUSUC_ImageNet_merge(bias, Y_correct_S, Y_correct_U, class_correct_S, class_count_S, class_count_U, loc_S, loc_U, N_S, N_U, fixed_bias);
        
        save(['../EXEM_results/EXEM' split_or_not opt.GZSL '_' opt.test_type '_ImageNet_' task '_' feature_name semantic_name '_' norm_method ...
            '_flat' num2str(flat_hit(p)) '.mat'], 'AUC_val_class_wise', 'AUC_record_class_wise', 'acc_noBias_class_wise', 'acc_class_wise', 'HM_class_wise',...
            'AUC_val_sample_wise', 'AUC_record_sample_wise', 'acc_noBias_sample_wise', 'acc_sample_wise', 'HM_sample_wise');
        disp([flat_hit(p), AUC_val_class_wise, AUC_val_sample_wise]);
    end
    
end

end

function [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(acc_mat)
    [~, max_ind] = max(acc_mat(:));
    [loc_C, loc_nu, loc_gamma, loc_pca_d] = ind2sub(size(acc_mat), max_ind);
end
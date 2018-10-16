function EXEM(task, test_type, dataset, opt, direct_test, feature_name)

%% Input
% task: 'train', 'test'
% test_type: 'dis_eu', 'dis_seu', 'acc_eu', 'acc_seu'
% dataset: 'AWA', 'CUB', 'SUN'
% opt: opt.C: the regularizer coefficient of nu-SVR (e.g, 2 .^ 0)
%      opt.nu: the nu-SVR parameter (e.g, 2 .^ (-10 : 0))
%      opt.gamma: the RBF scale parameter (e.g., 2 .^ (-5 : 5))
%      opt.pca_d: the PCA dimensionality (e.g., 500)
%      opt.ind_split: AWA: []; CUB: choose one from 1:4; SUN: choose one from 1:10
% direct_test: test on a specific [C, nu, gamma, pca_d] without cross-validation

%% Settings
set_path_EXEM;
norm_method = 'L2';

%% Data
[Xtr, Ytr, Xte, Yte, attr2, class_order] = data_loader(dataset, opt, feature_name, 'not');

nr_fold = 5;
Sig_Y = get_class_signatures(attr2, norm_method);

%% 5-fold class-wise cross validation splitting (for 'train' and 'val')
fold_loc = cv_split(task, Ytr, class_order);

%% training & validation
if (strcmp(task, 'train'))
    % record for validation
    val_dis_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d));
    val_dis_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d));
    val_acc_eu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d));
    val_acc_seu = zeros(length(opt.C), length(opt.nu), length(opt.gamma), length(opt.pca_d));
    
    for f = 1 : nr_fold
        Xbase = Xtr;
        Xbase(fold_loc{f}, :) = [];
        Ybase = Ytr;
        Ybase(fold_loc{f}) = [];
        Xval = Xtr(fold_loc{f}, :);
        Yval = Ytr(fold_loc{f});
        Sig_Ybase = Sig_Y(unique(Ybase), :);
        Sig_Yval = Sig_Y(unique(Yval), :);
        
        %% Starting of EXEM training
        % PCA learning
        [mean_Xbase_PCA, V] = do_pca(Xbase);
        Xbase = bsxfun(@minus, Xbase, mean_Xbase_PCA);
        Xval = bsxfun(@minus, Xval, mean_Xbase_PCA);
        
        for g = 1: length(opt.gamma)
            % SVR kernel
            Ker_base = [(1 : length(unique(Ybase)))', exp(-opt.gamma(g) * pdist2_fast(Sig_Ybase, Sig_Ybase) .^ 2)];
            Ker_val = [(1 : length(unique(Yval)))', exp(-opt.gamma(g) * pdist2_fast(Sig_Yval, Sig_Ybase) .^ 2)];

            for d = 1 : length(opt.pca_d)
                disp([f, g, d])
                % PCA projection
                mapped_Xbase = Xbase * V(:, 1 : opt.pca_d(d));
                mapped_Xval = Xval * V(:, 1 : opt.pca_d(d));
                [mean_Xbase, std_Xbase] = compute_class_stat(Ybase, mapped_Xbase);
                avg_std_Xbase = mean(std_Xbase, 1);
                
                for c = 1 : length(opt.C)
                    for n = 1 : length(opt.nu)
                        cur_C = opt.C(c); cur_nu = opt.nu(n);
                        mean_Xbase_rec = zeros(size(Sig_Ybase, 1), opt.pca_d(d));
                        mean_Xval_rec = zeros(size(Sig_Yval, 1), opt.pca_d(d));
                        mean_Xval = zeros(size(Sig_Yval, 1), opt.pca_d(d));

                        parfor j = 1 : opt.pca_d(d)
                            % SVR learning and testing
                            regressor = svmtrain(mean_Xbase(:, j), Ker_base, ['-s 4 -t 4 -c ' num2str(cur_C) ' -n ' num2str(cur_nu) ' -m 10000']);
                            mean_Xbase_rec(:, j) = svmpredict(mean_Xbase(:, j), Ker_base, regressor);
                            mean_Xval_rec(:, j) = svmpredict(mean_Xval(:, j), Ker_val, regressor);
                        end
                        
                        [dis_eu, dis_seu, acc_eu, acc_seu] = test_EXEM(mapped_Xval, Yval, mean_Xval_rec, avg_std_Xbase);
                        val_dis_eu(c, n, g, d) = val_dis_eu(c, n, g, d) + dis_eu / nr_fold;
                        val_dis_seu(c, n, g, d) = val_dis_seu(c, n, g, d) + dis_seu / nr_fold;
                        val_acc_eu(c, n, g, d) = val_acc_eu(c, n, g, d) + acc_eu / nr_fold;
                        val_acc_seu(c, n, g, d) = val_acc_seu(c, n, g, d) + acc_seu / nr_fold;
                    end
                end
            end
        end
    end
    
    save(['../EXEM_CV_results/EXEM_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '.mat'],...
    'val_dis_eu', 'val_dis_seu', 'val_acc_eu', 'val_acc_seu', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    if(isempty(direct_test) || length(direct_test) == 1)
        load(['../EXEM_CV_results/EXEM_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '.mat'],...
            'val_dis_eu', 'val_dis_seu', 'val_acc_eu', 'val_acc_seu', 'opt');
        if (strcmp(test_type, 'dis_eu'))
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(-val_dis_eu, direct_test);
        elseif (strcmp(test_type, 'dis_seu'))
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(-val_dis_seu, direct_test);
        elseif (strcmp(test_type, 'acc_eu'))
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(val_acc_eu, direct_test);
        elseif (strcmp(test_type, 'acc_seu'))
            [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(val_acc_seu, direct_test);
        else
            disp('Wrong test type!');
            return;
        end
        C = opt.C(loc_C(1)); nu = opt.nu(loc_nu(1)); gamma = opt.gamma(loc_gamma(1)); pca_d = opt.pca_d(loc_pca_d(1));
        disp([loc_C(1), loc_nu(1), loc_gamma(1), loc_pca_d(1)]);
    else
        C = direct_test(1); nu = direct_test(2); gamma = direct_test(3); pca_d = direct_test(4);
    end
    
    Sig_Ytr = Sig_Y(unique(Ytr), :);
    Sig_Yte = Sig_Y(unique(Yte), :);
    
    %% Starting of EXEM training
    [mean_Xtr_PCA, V] = do_pca(Xtr);
    Xtr = bsxfun(@minus, Xtr, mean_Xtr_PCA);
    Xte = bsxfun(@minus, Xte, mean_Xtr_PCA);
    
    % SVR kernel
    Ker_tr = [(1 : length(unique(Ytr)))', exp(-gamma * pdist2_fast(Sig_Ytr, Sig_Ytr) .^ 2)];
    Ker_te = [(1 : length(unique(Yte)))', exp(-gamma * pdist2_fast(Sig_Yte, Sig_Ytr) .^ 2)];
    
    % PCA projection
    mapped_Xtr = Xtr * V(:, 1 : pca_d);
    mapped_Xte = Xte * V(:, 1 : pca_d);
    [mean_Xtr, std_Xtr] = compute_class_stat(Ytr, mapped_Xtr);
    avg_std_Xtr = mean(std_Xtr, 1);

    mean_Xtr_rec = zeros(size(Sig_Ytr, 1), pca_d);
    mean_Xte_rec = zeros(size(Sig_Yte, 1), pca_d);
    mean_Xte = zeros(size(Sig_Yte, 1), pca_d);
    regressors = cell(1, pca_d);
    
    parfor j = 1 : pca_d
        % SVR learning and testing
        regressors{j} = svmtrain(mean_Xtr(:, j), Ker_tr, ['-s 4 -t 4 -c ' num2str(C) ' -n ' num2str(nu) ' -m 10000']);
        mean_Xtr_rec(:, j) = svmpredict(mean_Xtr(:, j), Ker_tr, regressors{j});
        mean_Xte_rec(:, j) = svmpredict(mean_Xte(:, j), Ker_te, regressors{j});
    end
                        
    [dis_eu, dis_seu, acc_eu, acc_seu] = test_EXEM(mapped_Xte, Yte, mean_Xte_rec, avg_std_Xtr);
    disp([dis_eu, dis_seu, acc_eu, acc_seu]);
    
    attr2 = zeros(size(attr2, 1), pca_d);
    attr2(unique(Ytr), :) = mean_Xtr_rec;
    attr2(unique(Yte), :) = mean_Xte_rec;
    
    if (strcmp(test_type, 'dis_eu') || strcmp(test_type, 'dis_seu')) 
        save(['../EXEM_results/attr_EXEM_' dataset '_' test_type '_split' num2str(opt.ind_split) '_' feature_name '_'...
            norm_method '_C' num2str(C) '_nu' num2str(nu) '_gamma' num2str(gamma) '_pca_d' num2str(pca_d) '.mat']...
            , 'C', 'nu', 'gamma', 'pca_d', 'attr2');
    
    else
        save(['../EXEM_results/EXEM_' dataset '_' test_type '_split' num2str(opt.ind_split) '_' feature_name '_'...
            norm_method '_C' num2str(C) '_nu' num2str(nu) '_gamma' num2str(gamma) '_pca_d' num2str(pca_d) '.mat']...
            , 'regressors', 'C', 'nu', 'gamma', 'pca_d', 'dis_eu', 'dis_seu', 'acc_eu', 'acc_seu');
    end
end
end

function [loc_C, loc_nu, loc_gamma, loc_pca_d] = find_max(acc_mat, direct_test)
    if length(direct_test) == 1
        loc_pca_d = direct_test;
        acc_mat = acc_mat(:, :, :, loc_pca_d);
        [~, max_ind] = max(acc_mat(:));
        [loc_C, loc_nu, loc_gamma] = ind2sub(size(acc_mat), max_ind);
    else
        [~, max_ind] = max(acc_mat(:));
        [loc_C, loc_nu, loc_gamma, loc_pca_d] = ind2sub(size(acc_mat), max_ind);
    end
end
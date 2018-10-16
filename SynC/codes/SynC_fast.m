function SynC_fast(task, dataset, opt, direct_test, feature_name)

%% Input
% task: 'train', 'val', 'test'
% dataset: 'AWA', 'CUB', 'SUN'
% opt: opt.lambda: the regularizer coefficient on W in training (e.g, 2 .^ (-24 : -9))
%      opt.Sim_scale: the RBF scale parameter for computing semantic similarities (e.g., 2 .^ (-5 : 5))
%      opt.ind_split: AWA: []; CUB: choose one from 1:4; SUN: choose one from 1:10
%      opt.loss_type: 'OVO', 'CS', 'struct'
% direct_test: test on a specific [lambda, Sim_scale] pair without cross-validation

%% Settings
set_path_SynC;
norm_method = 'L2'; Sim_type = 'RBF_norm';

%% Data
[Xtr, Ytr, Xte, Yte, attr2, class_order] = data_loader(dataset, opt, feature_name, 'not'); % not EXEM(SynC)

nr_fold = 5;
Sig_Y = get_class_signatures(attr2, norm_method);
Sig_dist = Sig_dist_comp(Sig_Y);

%% 5-fold class-wise cross validation splitting (for 'train' and 'val')
fold_loc = cv_split(task, Ytr, class_order);

%% training
if (strcmp(task, 'train'))
    for i = 1 : length(opt.lambda)
        W_record = cell(1, nr_fold);
        for j = 1 : nr_fold
            Xbase = Xtr;
            Xbase(fold_loc{j}, :) = [];
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            
            if (strcmp(opt.loss_type, 'OVO'))
                W = train_W_OVO([], Xbase, Ybase, opt.lambda(i));
            elseif (strcmp(opt.loss_type, 'CS'))
                W = train_W_CS([], Xbase, Ybase, opt.lambda(i));
            elseif (strcmp(opt.loss_type, 'struct'))
                W = train_W_struct([], Xbase, Ybase, Sig_dist(unique(Ybase), unique(Ybase)), opt.lambda(i));
            else
                disp('Wrong loss type!');
                return;
            end
            W_record{j} = W;

            save(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type...
                '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        end
    end
end

%% validation
if (strcmp(task, 'val'))
    
    
    acc_val = zeros(length(opt.lambda), length(opt.Sim_scale));
    for i = 1 : length(opt.lambda)
        
        
        
        
        
        
        
        
        load(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type...
            '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        
        for j = 1 : nr_fold
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            Xval = Xtr(fold_loc{j}, :);
            Yval = Ytr(fold_loc{j});
            W = W_record{j};

            for k = 1 : length(opt.Sim_scale)
                Sim_base = Compute_Sim(Sig_Y(unique(Ybase), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);
                Sim_val = Compute_Sim(Sig_Y(unique(Yval), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);
                V = pinv(Sim_base) * W;            
                Ypred_val = test_V(V, Sim_val, Xval, Yval);
                acc_val(i, k) = acc_val(i, k) + evaluate_easy(Ypred_val, Yval) / nr_fold;          
            end
            clear W;
        end
        clear W_record;
    end
    save(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type '.mat'],...
        'acc_val', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    if(isempty(direct_test))
        load(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type '.mat'],...
            'acc_val', 'opt');
        [loc_lambda, loc_Sim_scale] = find(acc_val == max(acc_val(:)));
        lambda = opt.lambda(loc_lambda(1)); Sim_scale = opt.Sim_scale(loc_Sim_scale(1));
        disp([loc_lambda(1), loc_Sim_scale(1)])
    else
        lambda = direct_test(1); Sim_scale = direct_test(2);
    end
    
    if (exist(['../SynC_results/SynC_fast_' opt.loss_type '_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type...
        '_lambda' num2str(lambda) '_Sim_scale' num2str(Sim_scale) '.mat'], 'file') == 2)
        
        disp('load existing file!!');
        load(['../SynC_results/SynC_fast_' opt.loss_type '_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type...
        '_lambda' num2str(lambda) '_Sim_scale' num2str(Sim_scale) '.mat'], 'W');
    
    else
        disp('train a new model!!');
        if (strcmp(opt.loss_type, 'OVO'))
            W = train_W_OVO([], Xtr, Ytr, lambda);
        elseif (strcmp(opt.loss_type, 'CS'))
            W = train_W_CS([], Xtr, Ytr, lambda);
        elseif (strcmp(opt.loss_type, 'struct'))
            W = train_W_struct([], Xtr, Ytr, Sig_dist(unique(Ytr), unique(Ytr)), lambda);
        else
            disp('Wrong loss type!');
            return;
        end
    end
    
    Sim_tr = Compute_Sim(Sig_Y(unique(Ytr), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    Sim_te = Compute_Sim(Sig_Y(unique(Yte), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    V = pinv(Sim_tr) * W;
    Ypred_te = test_V(V, Sim_te, Xte, Yte);
    acc_te = evaluate_easy(Ypred_te, Yte);
    disp(acc_te);

    save(['../SynC_results/SynC_fast_' opt.loss_type '_' dataset '_split' num2str(opt.ind_split) '_' feature_name '_' norm_method '_' Sim_type...
        '_lambda' num2str(lambda) '_Sim_scale' num2str(Sim_scale) '.mat'], 'W', 'V', 'lambda', 'Sim_scale', 'acc_te');
end

end

function Sig_dist = Sig_dist_comp(Sig_Y)
inner_product = Sig_Y * Sig_Y';
C = size(Sig_Y, 1);
Sig_dist = max(diag(inner_product) * ones(1, C) + ones(C, 1) * diag(inner_product)' - 2 * inner_product, 0);
Sig_dist = sqrt(Sig_dist);
end
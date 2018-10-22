%% Introduction
% EXEM: EXEM
% EXEM on GZSL: EXEM_GZSL

%% EXEM Demo for 'AWA'
datasets = {'AWA', 'CUB', 'SUN', 'AWA1_SS', 'AWA1_PS', 'AWA2_SS', 'AWA2_PS', 'CUB_SS', 'CUB_PS', 'SUN_SS', 'SUN_PS'};
i = 5;

%% Path
cd ./EXEM/codes

%% Parameters setting
test_type = 'acc_seu'; % or 'acc_eu'

if i > 3
    opt.C = 2 .^ (-5 : 5);
    opt.nu = 2 .^ (-8 : 0);
    opt.gamma = 2 .^ (-4 : 4);
    opt.pca_d = [500, 1000];
    feature_type = 'resnet';
else
    opt.C = 2 .^ (-3 : 3);
    opt.nu = 2 .^ (-8 : 0);
    opt.gamma = 2 .^ (-4 : 4);
    opt.pca_d = 500;
    feature_type = 'googleNet';
end

if strcmp(datasets{i}, 'CUB') || strcmp(datasets{i}, 'SUN')
    opt.ind_split = 1; % for 'CUB', from 1 to 4; for 'SUN' from 1 to 10; need to take average performance finally
else
    opt.ind_split = [];
end

%% Training, validation, testing
disp ('Cross-validation');
EXEM('train', test_type, datasets{i}, opt, [], feature_type);

disp ('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
EXEM('test', test_type, datasets{i}, opt, [], feature_type);

disp ('You can also directly train a model and test, given pre-defined parameters');
cd ..
cd ..
% C = 2 ^ -3;
% nu = 2 ^ 0;
% gamma = 2 ^ -1;
% pca_d = 500;
% EXEM('test', test_type, 'AWA', opt, [C, nu, gamma, pca_d], feature_type);

%% GZSL
% test_type = 'acc_seu'; % or 'acc_eu' % or 'HM_eu'/'HM_seu' if using 'HM' for hyper-parameter tuning, which is never used in the paper
% disp ('Cross-validation');
% EXEM_GZSL('train', test_type, datasets{i}, opt, [], feature_type);
% 
% disp ('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
% EXEM_GZSL('test', test_type, datasets{i}, opt, [], feature_type);
% 
% disp ('You can also directly train a model and test, given pre-defined parameters');
% C = 2 ^ -3;
% nu = 2 ^ 0;
% gamma = 2 ^ -1;
% pca_d = 500;
% fixed_bias = ;
% EXEM_GZSL('test', test_type, 'AWA', opt, [C, nu, gamma, pca_d, fixed_bias], feature_type);
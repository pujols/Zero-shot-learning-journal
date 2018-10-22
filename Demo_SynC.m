%% Introduction
% SynC: SynC_fast
% EXEM(SynC): SynC_fast_EXEM

% SynC on GZSL: SynC_fast_GZSL
% EXEM(SynC) on GZSL: SynC_fast_EXEM_GZSL

%% SynC_fast Demo for 'AWA'
datasets = {'AWA', 'CUB', 'SUN', 'AWA1_SS', 'AWA1_PS', 'AWA2_SS', 'AWA2_PS', 'CUB_SS', 'CUB_PS', 'SUN_SS', 'SUN_PS'};
i = 5;

%% Path
cd ./SynC/codes

%% Parameters setting
opt.loss_type = 'OVO'; % 'CS', 'struct'

if i > 3
    opt.lambda = 2 .^ (-30 : 2 : -8);
    opt.Sim_scale = 2 .^ (-5 : 5);
    feature_type = 'resnet';
else
    opt.lambda = 2 .^ (-24 : -9);
    opt.Sim_scale = 2 .^ (-5 : 5);
    feature_type = 'googleNet';
end

if strcmp(datasets{i}, 'CUB') || strcmp(datasets{i}, 'SUN')
    opt.ind_split = 1; % for 'CUB', from 1 to 4; for 'SUN' from 1 to 10; need to take average performance finally
else
    opt.ind_split = [];
end

%% Training, validation, testing
disp ('Training in cross-validation');
SynC_fast('train', datasets{i}, opt, [], feature_type);

disp ('Collecting validation results');
SynC_fast('val', datasets{i}, opt, [], feature_type);

disp ('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
SynC_fast('test', datasets{i}, opt, [], feature_type);

disp ('You can also directly train a model and test, given a pair of selected lambda and Sim_scale');
cd ..
cd ..
% lambda = 2 ^ -10;
% Sim_scale = 2 ^ 0;
% SynC_fast('test', 'AWA', opt, [lambda, Sim_scale], feature_type);

%% SynC_fast_EXEM; i.e., EXEM(SynC)
% % You can change 'SynC_fast' to 'SynC_fast_EXEM'
% % Note that for opt.loss_type = 'OVO' or 'CS', you don't need to re-do
% % SynC_fast_EXEM('train', datasets{i}, opt, [], feature_type) but can
% % directly do SynC_fast_EXEM('val', datasets{i}, opt, [], feature_type)

%% GZSL
% test_type = []; % 'HM' if using 'HM' for hyper-parameter tuning, which is never used in the paper
% disp ('Training in cross-validation');
% SynC_fast_GZSL('train', test_type, datasets{i}, opt, [], feature_type);
% 
% disp ('Collecting validation results');
% SynC_fast_GZSL('val', test_type, datasets{i}, opt, [], feature_type);
% 
% disp ('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
% SynC_fast_GZSL('test', test_type, datasets{i}, opt, [], feature_type);
% 
% disp ('You can also directly train a model and test, given a pair of selected lambda and Sim_scale');
% lambda = 2 ^ -10;
% Sim_scale = 2 ^ 0;
% fixed_bias = ;
% SynC_fast_GZSL('test', test_type, 'AWA', opt, [lambda, Sim_scale, fixed_bias], feature_type);

% % You can change 'SynC_fast_GZSL' to 'SynC_fast_EXEM_GZSL'
% % Note that for opt.loss_type = 'OVO' or 'CS', you don't need to re-do
% % SynC_fast_EXEM_GZSL('train', test_type, datasets{i}, opt, [], feature_type) but can
% % directly do SynC_fast_EXEM_GZSL('val', test_type, datasets{i}, opt, [], feature_type)
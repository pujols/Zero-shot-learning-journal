%% EXEM Demo for AWA
disp('Training in cross-validation');
opt.C = 2 .^ (-5 : 5);
opt.nu = 2 .^ (-8 : 0);
opt.gamma = 2 .^ (-4 : 4);
opt.pca_d = 500;

%% ###############################################
opt.ind_split = [];
EXEM('train', [], 'AWA', opt, []);

disp('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
SynC_fast('test', 'acc_seu', 'AWA', opt, []);

% disp('You can also directly train a model and test, given a pair of selected lambda and Sim_scale');
% C = 2 ^ -3;
% nu = 2 ^ 0;
% gamma = 2 ^ -1;
% pca_d = 500;
% SynC_fast('test', 'AWA', opt, [C, nu, gamma, pca_d]);
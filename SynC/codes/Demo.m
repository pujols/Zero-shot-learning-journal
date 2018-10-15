% We provide three versions SynC, SynC_fast, and SynC_regV
% SynC and SynC_fast should provide similar results (the results in our CVPR 2016 paper are by SynC_fast).
% SynC_fast is much faster than SynC and SynC_regV in cross-validation.
% SynC_regV usually performs better by >2%.

%% SynC_fast Demo for AWA
display('Training in cross-validation');
opt.lambda = 2 .^ (-12 : -9);
opt.Sim_scale = 2 .^ (-2 : 2);
%% ################## IMPORTANT ##################
% Note that, the above range of "hyper-parameters" is for "Demo" only.
% The range used in our experiments for AwA, CUB, and SUN in SynC_fast.m is:
% opt.lambda = 2 .^ (-24 : -9);
% opt.Sim_scale = 2 .^ (-5 : 5);
% Please also check the instruction in SynC.m, SynC_fast.m, and SynC_regV.m for the setting of hyper-parameters and inputs.
%% ###############################################
opt.ind_split = [];
opt.loss_type = 'OVO';
SynC_fast('train', 'AWA', opt, []);

display('Collecting validation results');
SynC_fast('val', 'AWA', opt, []);

display('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
SynC_fast('test', 'AWA', opt, []);

display('You can also directly train a model and test, given a pair of selected lambda and Sim_scale');
lambda = 2 ^ -10;
Sim_scale = 2 ^ 0;
SynC_fast('test', 'AWA', opt, [lambda, Sim_scale]);
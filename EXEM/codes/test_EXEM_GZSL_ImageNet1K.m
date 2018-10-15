function [score_S_eu, score_U_eu, score_S_seu, score_U_seu] = test_EXEM_GZSL_ImageNet1K(X, mean_S_rec, mean_U_rec, std_Xtr)

% Prediction accuracy
score_S_eu = test_NN_euclidean(X, mean_S_rec);
score_U_eu = test_NN_euclidean(X, mean_U_rec);

score_S_seu = test_NN_std_euclidean(X, mean_S_rec, std_Xtr);
score_U_seu = test_NN_std_euclidean(X, mean_U_rec, std_Xtr);
end

%%%%%%%%%%%%%%%%%%%% Loss or Accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%
function score = test_NN_euclidean(X, mean_X)
score = -pdist2_fast(X, mean_X);
end

function score = test_NN_std_euclidean(X, mean_X, std_X)
score = -pdist2_fast(bsxfun(@rdivide, X, std_X), bsxfun(@rdivide, mean_X, std_X));
end
function [dis_eu, dis_seu, acc_eu, acc_seu] = test_EXEM(X, Y, mean_X_rec, std_Xtr)

[mean_X, std_X] = compute_class_stat(Y, X);

% Reconstruction loss
dis_eu = get_loss_mean(mean_X_rec, mean_X);
dis_seu = get_loss_mean_std(mean_X_rec, mean_X, std_X);

% Prediction accuracy
acc_eu = test_NN_euclidean(X, mean_X_rec, Y);
acc_seu = test_NN_std_euclidean(X, mean_X_rec, std_Xtr, Y);
end

%%%%%%%%%%%%%%%%%%%% Loss or Accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%
function acc = test_NN_euclidean(X, mean_X, Y)
labelSet = unique(Y);
pwdist = pdist2_fast(X, mean_X);
[~, Ypred] = min(pwdist, [], 2);
Ypred = labelSet(Ypred);
acc = evaluate_easy(Ypred, Y);
end

function acc = test_NN_std_euclidean(X, mean_X, std_X, Y)
labelSet = unique(Y);
pwdist = pdist2_fast(bsxfun(@rdivide, X, std_X), bsxfun(@rdivide, mean_X, std_X));
[~, Ypred] = min(pwdist, [], 2);
Ypred = labelSet(Ypred);
acc = evaluate_easy(Ypred, Y);
end

function loss = get_loss_mean(x_rec, x)
loss = x - x_rec;
loss = mean(sqrt(sum(loss .^ 2, 2)));
end

function loss = get_loss_mean_std(x_rec, x, std_x)
loss = (x - x_rec) ./ std_x;
loss =  mean(sqrt(sum(loss .^ 2, 2)));
end
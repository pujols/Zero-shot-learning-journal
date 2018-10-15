function dis_eu = test_EXEM_ImageNet1K(X, Y, mean_X_rec)

[mean_X, ~] = compute_class_stat(Y, X);
dis_eu = get_loss_mean(mean_X_rec, mean_X);
end

%%%%%%%%%%%%%%%%%%%% Loss or Accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%
function loss = get_loss_mean(x_rec, x)
loss = x - x_rec;
loss = mean(sqrt(sum(loss .^ 2, 2)));
end
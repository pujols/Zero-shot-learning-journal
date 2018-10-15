function [mean_X, V] = do_pca(X)

mean_X = mean(X, 1);
norm_X = bsxfun(@minus, X, mean_X);
cov_X = norm_X' * norm_X;
[V, D] = eig((cov_X + cov_X') / 2);
V = bsxfun(@rdivide, V, sqrt(sum(V .^ 2, 1)));
V(isnan(V)) = 0; V(isinf(V)) = 0;
D = diag(D);
[~, loc] = sort(D, 'descend');
V = V(:, loc);

end
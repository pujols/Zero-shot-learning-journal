function [W, flag] = train_W_struct(W0, X, Y, sig_dis, lambda)

%% interface description:
% Input:
%   W0: a C-by-d matrix, the initialization of W
%   X: a N-by-d matrix, with each row a training instance
%   Y: a N-by-1 vctor, with each element the class label (1,...,C) of the corresponding instance
%   lambda: balance coefficient

% Output:
%   W: a C-by-d matrix, with each row the 1-vs-all classifier of the seen classes

%% pre-settig:
[N, d] = size(X);
labels = unique(Y);
C = length(labels);
ind = zeros(N, C);
y_loc = zeros(N, 1);
for i = 1 : C
    ind(Y == labels(i), i) = 1;
    y_loc(Y == labels(i), 1) = i;
end

if ((size(sig_dis, 1) ~= size(sig_dis, 2)) || (size(sig_dis, 1) ~= C))
    display('Error: train_W_struct');
    return;
end

%% Parameter initialization
options.Display = 'off';
options.Method = 'lbfgs';
options.optTol = 1e-10;
options.progTol = 1e-10;
options.MaxIter = 1000;
options.MaxFunEvals = 1200;

%% Begin training
if isempty(W0)
    W0 = randn(C, d) / 100;
end
funObj = @(arg)compute_fg_W(arg, X, ind, y_loc, sig_dis, lambda);
[W, ~, flag] = minFunc(funObj, W0(:), options);
W = reshape(real(W), [C, d]);

end


function [f, g] = compute_fg_W(W0, X, ind, y_loc, sig_dis, lambda)

[N, d] = size(X);
C = size(ind, 2);

W = reshape(real(W0), [C, d]);
XW = X * W';
y_index = N * (y_loc - 1) + (1 : N)';
XW_star = XW(y_index);
diff_XW = bsxfun(@minus, XW + sig_dis(y_loc, :), XW_star);
[val, loc] = max(diff_XW, [], 2);
max_index = N * (loc - 1) + (1 : N)';
diff_ind = -ind;
diff_ind(max_index) = diff_ind(max_index) + 1;
L = max(0, val);
f = sum(L) / N + lambda / 2 * sum(sum(W .^ 2));
g = (diff_ind' * X)/ N + lambda * W;
g = real(g(:));

end
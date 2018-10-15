function [W, flag] = train_W_OVO(W0, X, Y, lambda)

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
ind = -ones(N, C);

for i = 1 : C
    ind(Y == labels(i), i) = 1;
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
funObj = @(arg)compute_fg_W(arg, X, ind, lambda);
[W, ~, flag] = minFunc(funObj, W0(:), options);
W = reshape(real(W), [C, d]);

end


function [f, g] = compute_fg_W(W0, X, ind, lambda)

[N, d] = size(X);
C = size(ind, 2);

W = reshape(real(W0), [C, d]);
XW = X * W';
L = max(0, 1 - ind .* XW) .^ 2;
SV = double(L > eps);
f = sum(sum(L)) / N / C + lambda / 2 * sum(sum(W .^ 2));
g = 2 * ((XW - ind) .* SV)' * X / N / C + lambda * W;
g = real(g(:));

end
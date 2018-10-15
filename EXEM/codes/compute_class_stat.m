function [mean_X, std_X] = compute_class_stat(Y, X)

uY = unique(Y);
mean_X = zeros(length(uY), size(X, 2));
std_X = zeros(length(uY), size(X, 2));
for i = 1 : length(uY)
    mean_X(i, :) = mean(X(Y == uY(i), :), 1);
    std_X(i, :) = nanstd(X(Y == uY(i), :));
end

end
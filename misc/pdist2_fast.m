function dist = pdist2_fast(sig_Y, sig_R)

    % dist = sqrt(max(sum(sig_Y .^ 2, 2) * ones(1, size(sig_R, 1)) + ones(size(sig_Y, 1), 1) * sum(sig_R .^ 2, 2)' - 2 * sig_Y * sig_R', 0));
    dist = sqrt(max(repmat(sum(sig_Y .^ 2, 2), [1, size(sig_R, 1)]) + repmat(sum(sig_R .^ 2, 2)', [size(sig_Y, 1), 1]) - 2 * sig_Y * sig_R', 0));
end
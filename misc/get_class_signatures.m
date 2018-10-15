function [Sig_Y, median_Dist] = get_class_signatures(w2v, norm_method)

if (strcmp(norm_method, 'L2'))
    disp('L2 normalization');
    Sig_Y = bsxfun(@rdivide, w2v, sqrt(sum(w2v .^ 2, 2)));
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
elseif (strcmp(norm_method, 'L1'))
    disp('L1 normalization');
    Sig_Y = bsxfun(@rdivide, w2v, sum(abs(w2v), 2));
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
elseif (strcmp(norm_method, 'zscore'))
    disp('zscore normalization');
    Sig_Y = zscore(w2v);
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
elseif (strcmp(norm_method, 'L2_zscore'))
    disp('L2 zscore normalization');
    Sig_Y = bsxfun(@rdivide, w2v, sqrt(sum(w2v .^ 2, 2)));
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
    Sig_Y = zscore(Sig_Y);
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
elseif (strcmp(norm_method, 'L1_zscore'))
    disp('L1 zscore normalization');
    Sig_Y = bsxfun(@rdivide, w2v, sum(abs(w2v), 2));
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
    Sig_Y = zscore(Sig_Y);
    Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
else
    disp('No normalization');
    Sig_Y = w2v;
end

Dist = pdist2_fast(Sig_Y, Sig_Y);
median_Dist = median(Dist(Dist > 0));
Sig_Y = Sig_Y / median_Dist;
Sig_Y(isnan(Sig_Y)) = 0; Sig_Y(isinf(Sig_Y)) = 0;
end


function [Xtr, Ytr, Xhold, Yhold, Xte, Yte, CV_hold_ind, CV_hold_loc] = data_loader_hold_GZSL(dataset, feature_name, Xtr, Ytr, Xte, Yte, opt)

if isempty(strfind(dataset, '_PS'))
    % dataset: AWA, CUB, SUN, AWA1_SS, AWA2_SS, CUB_SS, SUN_SS
    load (['../../data/GZSL_info/' dataset '_GZSL_inform.mat'], 'tr_hold_ind', 'CV_hold_ind', 'CV_hold_loc');

    if(strcmp(dataset, 'CUB') || strcmp(dataset, 'SUN'))
        tr_hold_ind = tr_hold_ind{opt.ind_split};
        CV_hold_ind = CV_hold_ind{opt.ind_split};
        CV_hold_loc = CV_hold_loc(opt.ind_split, :);
    end
    tr_hold_ind = tr_hold_ind(:);
    CV_hold_ind = CV_hold_ind(:);

    Xte = [Xtr(tr_hold_ind, :); Xte];
    Yte = [Ytr(tr_hold_ind); Yte];
    Xtr(tr_hold_ind, :) = [];
    Ytr(tr_hold_ind) = [];
    Xhold = Xtr(CV_hold_ind, :);
    Yhold = Ytr(CV_hold_ind);
    Xtr(CV_hold_ind, :) = [];
    Ytr(CV_hold_ind) = [];

else
    % dataset: AWA1_PS, AWA2_PS, CUB_PS, SUN_PS
    load (['../../data/' dataset '_' feature_name '.mat'], 'X', 'Y', 'te_loc_seen');
    load (['../../data/GZSL_info/' dataset '_GZSL_inform.mat'], 'CV_hold_ind', 'CV_hold_loc');
    CV_hold_ind = CV_hold_ind(:);
    
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xte = [X(te_loc_seen, :); Xte];
    Yte = [Y(te_loc_seen); Yte];
    clear X; clear Y;
    Xhold = Xtr(CV_hold_ind, :);
    Yhold = Ytr(CV_hold_ind);
    Xtr(CV_hold_ind, :) = [];
    Ytr(CV_hold_ind) = [];

end
    
end
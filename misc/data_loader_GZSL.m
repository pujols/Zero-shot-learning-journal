function [Xtr, Ytr, Xte, Yte, attr2, class_order] = data_loader_GZSL(dataset, opt, feature_name, EXEM_or_not)

if(strcmp(dataset, 'AWA'))
    load (['../../data/AwA_' feature_name '.mat'], 'X', 'Y', 'attr2', 'tr_loc', 'te_loc', 'class_order'); attr2(attr2 == -1) = 0;
    if (strcmp(EXEM_or_not, 'yes'))
        load (['../../data/EXEM_info/AWA_EXEM_GZSL_' feature_name '.mat'], 'attr2');
    end
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X(tr_loc, :); Xte = X(te_loc, :); clear X;
    Ytr = Y(tr_loc); Yte = Y(te_loc); clear Y;
    
elseif(strcmp(dataset, 'CUB'))
    load (['../../data/CUB_' feature_name '.mat'], 'X', 'Y', 'attr2', 'CUB_class_loc', 'class_order');
    if (strcmp(EXEM_or_not, 'yes'))
        load (['../../data/EXEM_info/CUB_EXEM_GZSL_' feature_name '_split' num2str(opt.ind_split) '.mat'], 'attr2')
    end
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X; Xte = X(CUB_class_loc{opt.ind_split}, :); Xtr(CUB_class_loc{opt.ind_split}, :) = []; clear X;
    Ytr = Y; Yte = Y(CUB_class_loc{opt.ind_split}); Ytr(CUB_class_loc{opt.ind_split}) = []; clear Y;

elseif(strcmp(dataset, 'SUN'))
    load (['../../data/SUN_' feature_name '.mat'], 'X', 'Y', 'attr2', 'SUN_class_loc', 'class_order');
    if (strcmp(EXEM_or_not, 'yes'))
        load (['../../data/EXEM_info/SUN_EXEM_GZSL_' feature_name '_split' num2str(opt.ind_split) '.mat'], 'attr2')
    end
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X; Xte = X(SUN_class_loc{opt.ind_split}, :); Xtr(SUN_class_loc{opt.ind_split}, :) = []; clear X;
    Ytr = Y; Yte = Y(SUN_class_loc{opt.ind_split}); Ytr(SUN_class_loc{opt.ind_split}) = []; clear Y;
    class_order = class_order{opt.ind_split};
    
else
    % dataset: AWA1_PS, AWA2_PS, CUB_PS, SUN_PS (or _SS)
    load (['../../data/' dataset '_' feature_name '.mat'], 'X', 'Y', 'attr2', 'te_loc', 'tr_loc', 'class_order');
    if (strcmp(EXEM_or_not, 'yes'))
        load (['../../data/EXEM_info/' dataset '_EXEM_GZSL_' feature_name '.mat'], 'attr2')
    end
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X(tr_loc, :); Xte = X(te_loc, :); clear X;
    Ytr = Y(tr_loc); Yte = Y(te_loc); clear Y;
    
end

if isempty(strfind(dataset, 'SUN')) && ~strcmp(EXEM_or_not, 'yes')
    attr2 = attr2 / 100;
end

end
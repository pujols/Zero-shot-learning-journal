function Ypred = test_V(V, Sim, X, Y)

labelSet = unique(Y);
W = construct_W(V, Sim);
XW = X * W';
[~, Ypred] = max(XW, [], 2);
Ypred = labelSet(Ypred);
end

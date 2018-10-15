function [acc, confusion, acc2] = evaluate_easy(Ypred, Ytrue)

labels = unique(Ytrue);
L = length(labels);

confusion = zeros(L, 1);
for i = 1 : L
        confusion(i) = sum((Ytrue == labels(i)) & (Ypred == labels(i))) / sum(Ytrue == labels(i));
end

acc = mean(confusion);
acc2 = mean(Ypred == Ytrue);

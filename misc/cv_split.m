function fold_loc = cv_split(task, Ytr, class_order)

fold_loc = [];
if (strcmp(task, 'train') || strcmp(task, 'val'))
    nr_fold = 5;
    labelSet = unique(Ytr);
    labelSetSize = length(labelSet);
    fold_size = floor(labelSetSize / nr_fold);
    fold_loc = cell(1, nr_fold);
    

    for i = 1 : nr_fold
        for j = 1 : fold_size
            
            fold_loc{i} = [fold_loc{i}; find(Ytr == labelSet(class_order((i - 1) * fold_size + j)))];
        end
        
        fold_loc{i} = sort(fold_loc{i});
    end
end
end
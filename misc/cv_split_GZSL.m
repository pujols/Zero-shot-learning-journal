function [fold_loc, hold_fold_loc] = cv_split_GZSL(task, Ytr, class_order, CV_hold_ind, CV_hold_loc)

fold_loc = []; hold_fold_loc = []; 
if (strcmp(task, 'train') || strcmp(task, 'val'))
    nr_fold = 5;
    labelSet = unique(Ytr);
    labelSetSize = length(labelSet);
    fold_size = floor(labelSetSize / nr_fold);
    fold_loc = cell(1, nr_fold);
    hold_fold_loc = cell(1, nr_fold);

    for i = 1 : nr_fold
        for j = 1 : fold_size
            hold_fold_loc{i} = [hold_fold_loc{i}; find(ismember(CV_hold_ind, CV_hold_loc{class_order((i - 1) * fold_size + j)}))];
            fold_loc{i} = [fold_loc{i}; find(Ytr == labelSet(class_order((i - 1) * fold_size + j)))];        
        end
        hold_fold_loc{i} = sort(hold_fold_loc{i});
        fold_loc{i} = sort(fold_loc{i});
    end
end
end
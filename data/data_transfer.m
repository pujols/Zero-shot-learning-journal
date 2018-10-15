dataset = {'AWA1', 'AWA2', 'CUB', 'SUN'};

for i = 1 : length(dataset)
    disp(dataset{i});
    load (['./xlsa17/data/' dataset{i} '/res101.mat'], 'features', 'labels');
    X = features';
    Y = labels;
    num_class = length(unique(Y));

    %% PS
    load (['./xlsa17/data/' dataset{i} '/att_splits.mat'], 'original_att', 'trainval_loc', 'test_seen_loc', 'test_unseen_loc');
    if size(original_att, 1) ~= num_class       
        attr2 = original_att';
    else
        attr2 = original_att;
    end
    
    tr_loc = trainval_loc;
    te_loc = test_unseen_loc;
    te_loc_seen = test_seen_loc;

    load (['./class_info/' dataset{i} '_PS_class_info.mat'], 'class_order');
    save([dataset{i} '_PS_resnet.mat'], 'X', 'Y', 'attr2', 'tr_loc', 'te_loc', 'te_loc_seen', 'class_order');
    clear tr_loc; clear te_loc; clear te_loc_seen; clear class_order; clear original_att; clear trainval_loc; clear test_unseen_loc; clear test_seen_loc;
    
    %% SS
    load (['./standard_split/' dataset{i} '/att_splits.mat'], 'trainval_loc', 'test_unseen_loc');
    
    tr_loc = trainval_loc;
    te_loc = test_unseen_loc;

    load (['./class_info/' dataset{i} '_SS_class_info.mat'], 'class_order');
    save([dataset{i} '_SS_resnet.mat'], 'X', 'Y', 'attr2', 'tr_loc', 'te_loc', 'class_order');
    clear tr_loc; clear te_loc; clear class_order; clear trainval_loc; clear test_unseen_loc; 
    
    clear X; clear Y; clear features; clear attr2; clear labels;
end
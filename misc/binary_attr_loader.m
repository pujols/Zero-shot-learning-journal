function attr = binary_attr_loader(dataset)

if ~isempty(strfind(dataset, 'AWA'))
    load ../../data/binary_info/AWA_binary_attr_release.mat attr
    attr(attr == -1) = 0;
    
elseif (strcmp(dataset, 'CUB'))
    load ../../data/binary_info/CUB_binary_attr_release.mat attr

elseif (strcmp(dataset, 'SUN'))
    load ../../data/binary_info/SUN_binary_attr_release.mat attr
    
elseif ~isempty(strfind(dataset, 'CUB'))
    load ../../data/binary_info/CUB_resnet_binary_attr_release.mat attr
    
elseif ~isempty(strfind(dataset, 'SUN'))
    load ../../data/binary_info/SUN_resnet_binary_attr_release.mat attr
    
end
attr(attr ~= 0) = 1;
attr = attr(:, ((sum(attr, 1) ~= 0) & (sum(attr, 1) ~= size(attr, 1))));
end
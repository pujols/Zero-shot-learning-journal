function [AUC_val, AUC_record, acc_noBias, HM, fixed_bias] = Compute_AUSUC(score_S, score_U, Y, label_S, label_U, fixed_bias)

% score_S: #samples-by-#seen classes (columns should correspond to seen-class labels in ascending order)
% score_U: #samples-by-#unseen classes (columns should correspond to unseen-class labels in ascending order)
% Y: #samples-by-1 vector
% label_S: #classes-by-1 vector (in ascending order)
% label_U: #classes-by-1 vector (in ascending order)
% fixed_bias: a scalar of bias to increase unseen classes' scores. If [],
% the fixed_bias that leads to the highest HM will be returned 

AUC_record = zeros(length(Y) + 1, 2);
label_S = unique(label_S);
label_U = unique(label_U);

L_S = length(label_S);
L_U = length(label_U);
L = length(unique([label_S; label_U]));
if ((L_S + L_U ~= L) || (length(unique(Y)) ~= L))
    disp('Wrong seen-unseen separation');
    pause();
end
if ((L_S ~= size(score_S, 2)) || (L_U ~= size(score_U, 2)))
    disp('Wrong class number');
    pause();
end

%% effective bias searching
[max_S, loc_S] = max(score_S, [], 2);
Ypred_S = label_S(loc_S);
[max_U, loc_U] = max(score_U, [], 2);
Ypred_U = label_U(loc_U);
[class_correct_S, class_correct_U, class_count_S, class_count_U] = AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Y);
Y_correct_S = double(Ypred_S == Y);
Y_correct_U = double(Ypred_U == Y);
bias = max_S - max_U;
[bias, loc_B] = sort(bias);
[~, unique_bias_loc] = unique(bias); unique_bias_loc(unique_bias_loc == 1) = []; unique_bias_loc = unique_bias_loc - 1;
unique_bias_loc = unique([unique_bias_loc(:); length(bias)]);
bias = bias(unique_bias_loc);
%% efficient evaluation
acc_change_S = (Y_correct_S(loc_B) ./ class_count_S(loc_S(loc_B))) / L_S;
acc_change_U = (Y_correct_U(loc_B) ./ class_count_U(loc_U(loc_B))) / L_U;
AUC_record(:, 1) = [0; cumsum(-acc_change_S)] + mean(class_correct_S ./ class_count_S);
AUC_record(:, 2) = [0; cumsum(acc_change_U)];

if (sum(abs(AUC_record(end, :) - [0, mean(class_correct_U ./ class_count_U)])) > (10 ^ -12))
    disp('AUC wrong');
    pause();
end
AUC_record = AUC_record([0; unique_bias_loc(:)] + 1, :);
%% Compute AUC
acc_noBias = AUC_record(sum(bias <= 0) + 1, :);
AUC_val = trapz(AUC_record(:, 2), AUC_record(:, 1));
%% Compute Harmonic mean
if isempty(fixed_bias)
    HM = 2 * (AUC_record(:, 2) .* AUC_record(:, 1)) ./ (AUC_record(:, 2) + AUC_record(:, 1));
    [HM, fixed_bias_loc] = max(HM(:));
    fixed_bias_loc = max(1, fixed_bias_loc - 1);
    fixed_bias = bias(fixed_bias_loc);
else
    acc = AUC_record(sum(bias <= fixed_bias) + 1, :);
    HM = Compute_HM(acc);
    HM_nobias = Compute_HM(acc_noBias);
    disp(['without bias: acc_S: ' num2str(acc_noBias(1)) '; acc_U: ' num2str(acc_noBias(2)) '; HM: ' num2str(HM_nobias)]);
    disp(['with bias ' num2str(fixed_bias) ': acc_S: ' num2str(acc(1)) '; acc_U: ' num2str(acc(2)) '; HM: ' num2str(HM)]);
end

end

function HM = Compute_HM(acc)
HM = 2 * acc(1) * acc(2) / (acc(1) + acc(2));
end

function [class_correct_S, class_correct_U, class_count_S, class_count_U] = AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Ytrue)
L_S = length(label_S);
L_U = length(label_U);
class_count_S = zeros(L_S, 1);
class_count_U = zeros(L_U, 1);
class_correct_S = zeros(L_S, 1);
class_correct_U = zeros(L_U, 1);
for i = 1 : L_S
    class_count_S(i) = sum(Ytrue == label_S(i));
    class_correct_S(i) = sum((Ytrue == label_S(i)) & (Ypred_S == label_S(i)));
end
for i = 1 : L_U
    class_count_U(i) = sum(Ytrue == label_U(i));
    class_correct_U(i) = sum((Ytrue == label_U(i)) & (Ypred_U == label_U(i)));
end
class_count_S(class_count_S == 0) = 10 ^ 10;
class_count_U(class_count_U == 0) = 10 ^ 10;
end
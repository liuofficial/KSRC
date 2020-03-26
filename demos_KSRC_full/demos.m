clear; clc;

n = 1; % data
nt = 5; it = 1; % train and test data
alg = 1;

[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);

[train_idx, test_idx] = load_train_test(n, 1, nt, it);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);    
        
% parameters
mu = 1e-3; lam = 1e-4; wind = 5; 
sig = sqrt(0.5 ./ 40); sig0 = sqrt(0.5 ./ 0.7);
        
p.mu = mu; p.lam = lam;     

if alg == 1,
    % MF kernel
    [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, sig, [80 80], 1);
else
    % NF kernel
    [Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig, sig0, [80 80], 1);
end

AtX = Ktest; AtA = Ktrain;
S = SpRegKL1(AtX, AtA, p);
pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
test = Test.lab;
acc = class_eval(pred, test);

% OA
disp(acc.OA);
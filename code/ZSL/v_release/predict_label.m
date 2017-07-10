function CSCResult = predict_label(Opt, Data)
%Predict the labels of unseen instances
%input:  Opt parameters, Dataset
%output: the accuracy in Dataset
KESname  = Opt.KES.name{Opt.KESidx};
if Opt.useVWflag == true
    KESname  = Opt.vkesname{Opt.vKESidx};
end


Xtr = Data.TrainData; Xts = Data.TestData;
Ltr = Data.TrainLabel_c; Lts = Data.TestLabel_c;

k2cMat = double(Opt.KES.anchors);
cls_tr_id = Data.trainclasses_id;  
cls_ts_id = Data.testclasses_id;   
V = learn_fs_with_baseline(Xtr, Ltr, k2cMat, cls_tr_id, cls_ts_id, Opt.fs.gamma, Opt.fs.lambda);

Ktr = k2cMat(cls_tr_id,:); 
Kts = k2cMat(cls_ts_id,:); 
[acc_trp, acc_pc_trp, Otrp, Otr] = predict_csc_lm(Xtr, Ltr, V, Ktr, k2cMat, cls_tr_id,1);

[acc_tsp, acc_pc_tsp, Otsp, Ots] = predict_csc_lm(Xts, Lts, V, Kts, k2cMat, cls_ts_id,1);

[acc_t2tp, acc_pc_t2tp, Ot2tp, Ot2t] = predict_csc_lm([Xtr; Xts], [Ltr; Lts], V, [Ktr; Kts], k2cMat, [cls_tr_id; cls_ts_id],1);
[acc_s2tp, acc_pc_s2tp, Os2tp, Os2t] = predict_csc_lm(Xtr, Ltr, V, [Ktr; Kts], k2cMat, [cls_tr_id; cls_ts_id],1);
[acc_u2tp, acc_pc_u2tp, Ou2tp, Ou2t] = predict_csc_lm(Xts, Lts, V, [Ktr; Kts], k2cMat, [cls_tr_id; cls_ts_id],1);
fmtstr = {'%5.2f%%(cls_train)', '%5.2f%%(cls_validation)','%5.2f%%(cls_test)','%5.2f%%(cls_aug)',...
          '%5.2f%%(cls_T2T)',   '%5.2f%%(cls_S2T)',       '%5.2f%%(cls_U2T)'};
if ~isempty(acc_trp), fprintf(['      predict: ',fmtstr{1}], acc_trp); end

if ~isempty(acc_tsp), fprintf(['/', fmtstr{3}], acc_tsp); end

if ~isempty(acc_t2tp), fprintf(['/', fmtstr{5}], acc_t2tp); end
if ~isempty(acc_s2tp), fprintf(['/', fmtstr{6}], acc_s2tp); end
if ~isempty(acc_u2tp), fprintf(['/', fmtstr{7}], acc_u2tp); end
if ~isempty(acc_trp), fprintf('  |  '); end
CSCResult.acc_trp = max([acc_trp, 0]);
CSCResult.acc_tsp = max([acc_tsp, 0]);
CSCResult.acc_t2tp = max([acc_t2tp, 0]);
CSCResult.acc_s2tp = max([acc_s2tp, 0]);
CSCResult.acc_u2tp = max([acc_u2tp, 0]);
CSCResult.acc_pc_trp = acc_pc_trp;
CSCResult.acc_pc_tsp = acc_pc_tsp;
CSCResult.acc_pc_t2tp = acc_pc_t2tp;
CSCResult.acc_pc_s2tp = acc_pc_s2tp;
CSCResult.acc_pc_u2tp = acc_pc_u2tp;
end

function [acc_cls, acc_pc, Otxp, Otx] = predict_csc_lm(X, L, V, Sx, S, yidx, tK) %limited memory version
acc_cls = []; acc_pc = []; Otxp = []; Otx = [];
if ~isempty(X)
    Otx = X*V;
    A = Otx;
    for n = 1:size(A,1)
        A(n,:) = A(n,:)./norm(A(n,:));
    end
    Sx = S(yidx,:);
    Sx = Sx./repmat(sqrt(diag(Sx*Sx')),1, size(Sx,2));
    L_p = A*Sx';


    [~, Lidx] = sort(L_p, 2, 'descend');
    L_pred = Lidx(:, 1:tK);
    for i = 1:tK
        L_pred(:, i) = yidx(Lidx(:, i));
    end
    correctNum = 0; 
    for j = 1:length(L)
        correctNum = correctNum + ismember(L(j), L_pred(j,:));
    end
    acc_cls = 100*correctNum/length(L);
    
    acc_pc = accPerClass(L, L_pred(:,1));
    Otxp = S(L_pred(:,1),:);
end
end


function acc_pc = accPerClass(label, pred)
classes_id = sort(unique(label));
acc_pc = zeros(1, length(classes_id));
if isempty(pred) 
    return;
end
for i = 1 : length(classes_id)
    idx = find(label == classes_id(i));
    acc_pc(i) = 100*sum(pred(idx) == classes_id(i))/length(idx);
end
end
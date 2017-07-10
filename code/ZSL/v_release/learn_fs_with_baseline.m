function V = learn_fs_with_baseline(Xtr,Ltr,St, cls_tr_id, cls_ts_id, gamma, lambda)
Xtr_all = Xtr';
Ltr_all = Ltr;
Ztr_all = -1*ones(length(Ltr_all), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Ltr_all)
    c = Ltr_all(i);
    Ztr_all(i,c) = 1;
end
Ztr_all = Ztr_all(:,[cls_tr_id; cls_ts_id]);

Ztr = -1*ones(length(Ltr), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Ltr)
    c = Ltr(i);
    Ztr(i,c) = 1;
end
Ztr = Ztr(:,cls_tr_id);

S = St';
S = (S - min(S(:)))/(max(S(:))-min(S(:)));
S_tr = S(:, cls_tr_id);
% Tr, baseline model, ICML 2015, Torr
V0 = Xtr'*Xtr+gamma*eye(size(Xtr',1));
V1 = Xtr'*Ztr*S_tr';
V2 = V1/(S_tr*S_tr'+lambda*eye(size(S_tr,1)));
V  = V0\V2;

end
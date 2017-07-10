function [Btr, Bts] = DimNormalization(Atr, Ats)
% this function is used for normalizing visual feature V (samplenum*dim) or knowledge representation K (classnum*dim),
% Atr, Btr for seen matrix
% Ats, Bts for unseen matrix
dim_mean = mean(Atr, 1);
Btr = Atr - repmat(dim_mean, size(Atr,1), 1);
Bts = Ats - repmat(dim_mean, size(Ats,1), 1);
v_tr = max(abs(Btr), [], 1);
v_ts = v_tr;
for i = 1:length(v_tr)
    if v_tr(i) > 0
        Btr(:, i) = Btr(:, i)/v_tr(i);
    end
    if v_ts(i) > 0
        Bts(:, i) = Bts(:, i)/v_ts(i);
    end
end

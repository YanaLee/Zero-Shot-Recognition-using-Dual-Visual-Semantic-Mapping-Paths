function constructVKS(Opt, Data,U2T)
KESname  = Opt.KES.name{Opt.KESidx};
if Opt.useVWflag == true
    KESname  = Opt.vkesname{Opt.vKESidx};
end

Xtr = Data.TrainData;    Xts = Data.TestData;
Ltr = Data.TrainLabel_c;    Lts = Data.TestLabel_c;
Xtr_mu = Data.TrainDataMu;
k2cMat = double(Opt.KES.anchors);
cls_tr_id = Data.trainclasses_id;  
cls_ts_id = Data.testclasses_id;   
V = learn_fs_with_baseline(Xtr, Ltr, k2cMat, cls_tr_id, cls_ts_id, Opt.fs.gamma, Opt.fs.lambda);

kTop1 = Opt.kTop1;
kTop = Opt.kTop;
rTop = Opt.rTop;

genTrVKS = Data.TrainDataMu;

if U2T == 1
    genTsVKS = genvks(Xts, V, k2cMat, [cls_tr_id; cls_ts_id], kTop1, kTop, rTop);
    genTsVKS = genTsVKS(length(cls_tr_id)+1:length([cls_tr_id;cls_ts_id]),:);
else
    genTsVKS = genvks(Xts, V, k2cMat, cls_ts_id, kTop1, kTop, rTop);
end
genVKS = [genTrVKS; genTsVKS];



idx = [Opt.trainclasses_id; Opt.testclasses_id];
genVKS(idx,:) = genVKS;
K = genVKS;


save([Opt.path.knowledge Opt.dataset '_v_' Opt.featname{Opt.featidx} '.mat'], 'K');
end

function VKS = genvks(X, V, S, yidx, kTop1, kTop, rTop)
if ~isempty(X)
    A = X*V;
    for n = 1:size(A,1)
        A(n,:) = A(n,:)./norm(A(n,:));
    end    
    S = S./repmat(sqrt(diag(S*S')),1, size(S,2));
    Sx = S(yidx,:);
    Lp = A*Sx';
    
    nClass = length(yidx);
    VKS1 = zeros(nClass, size(V,1));
    VKS = zeros(nClass, size(V,1));
    
    %Plan A
    [DUMP, sidx] = sort(Lp, 1, 'descend');
    for i = 1:nClass
        x_idx = sidx(1:min(kTop1,size(sidx,1)), i);
        VKS1(i,:) = mean(X(x_idx, :));
    end
    %Plan B
    Lp_new = min(Lp(:))-ones(size(Lp));
    [DUMP, l_max] = max(Lp,[],2);
    l_num = zeros(nClass,1);
    for i = 1:nClass
        l_idx = find(l_max == i);
        l_num(i) = length(l_idx);
        Lp_new(l_idx, i) = Lp(l_idx, i);
    end
    [DUMP, sidx] = sort(Lp_new, 1, 'descend');
    K_pc = zeros(nClass, 1);
    for i = 1:nClass
        if l_num(i) > 0
            kTop_max = ceil(l_num(i)*rTop);
            %x_idx = sidx(1:l_num(i), i);
            x_idx = sidx(1:min(kTop, kTop_max), i);
            K_pc(i) = length(x_idx);
            VKS(i,:) = mean(X(x_idx, :));
        else
            VKS(i,:) = VKS1(i,:);
        end
    end
end
end

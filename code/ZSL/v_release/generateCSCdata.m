function [Opt, Data] = generateCSCdata(Opt, doPrjFlag, doNormalizeK)
% doPrjFlag = 0: no projection,  1-3: project Kt onto Ks with different constraints

featname = Opt.featname{Opt.featidx};
KESname  = Opt.KES.name{Opt.KESidx};
if Opt.useVWflag == true
    KESname = Opt.vkesname{Opt.vKESidx};
end

if doPrjFlag > 0
    prjstr = ['orthprj(', num2str(doPrjFlag), ')'];
else
    prjstr = 'no prj';
end
fprintf(['Generate CSC train and test data for ', Opt.dataset,' dataset (', featname, '-->', KESname, ') ....\n']);

%=======================================================================================
%  load semantic knowledge representation and pre-process
%=======================================================================================
load([Opt.path.knowledge, Opt.dataset, '_', KESname, '.mat']); % semantic knowledge
Opt.KES.dim = size(K, 2);
K = double(K);

if doNormalizeK == 1
    Ktr = K(Opt.trainclasses_id,:);
    Kts = K(Opt.testclasses_id,:);
    [Ktr_T, Kts_T] = DimNormalization(Ktr, Kts);
    K(Opt.trainclasses_id,:) = Ktr_T;
    K(Opt.testclasses_id,:) = Kts_T;
end
K = K./repmat(sqrt(diag(K*K')),1, size(K,2));      
%---------------------------------------------------------------------------------------
Opt.KES.anchors = K;
Y_kes = [];
y_cls = [];
for i=1:length(Opt.nperclass)
    y_cls = [y_cls; repmat(i,Opt.nperclass(i),1)];
    Y_kes = [Y_kes; repmat(Opt.KES.anchors(i,:),Opt.nperclass(i),1)];
end
Data.Y_kes = Y_kes;
Data.y_cls = y_cls;
Ktr = double(Opt.KES.anchors(Opt.trainclasses_id,:));
Kts = double(Opt.KES.anchors(Opt.testclasses_id,:));
if doPrjFlag > 0
    prjmodel = min(floor(doPrjFlag), 3);
    Beta = orthprj_reccoef(Opt, Ktr', Kts', prjmodel);
    Data.Kts_prj = (Ktr'*Beta)';
    Kts = Data.Kts_prj;  % Kts will be upgraded by new projected Kts. 
    Kts = Kts./repmat(sqrt(diag(Kts*Kts')),1, size(Kts,2));  % add by wdh, 2017-3-3        
end



Data.Ktr = Ktr;
Data.Kts = Kts;
Opt.KES.anchors(Opt.trainclasses_id,:) = Ktr;
Opt.KES.anchors(Opt.testclasses_id,:) = Kts;
Data.K = Opt.KES.anchors;   
%=======================================================================================


%=======================================================================================
%  load visual feature representation and pre-process
%=======================================================================================
load([Opt.path.feature, Opt.dataset, '_', featname, '.mat']); % semantic feature
Opt.featdim = size(X, 1);
X = double(X);
X = X';

TestData = [];
TestDataMu = []; % mean vector of each testing class;
TestLabel_K =[];
TestLabel_c = [];
TrainData = [];
TrainDataMu = []; % mean vector of each training class;
TrainLabel_K =[];
TrainLabel_c = [];
numpercls_tr = zeros(length(Opt.trainclasses_id),1);
for j=1:length(Opt.trainclasses_id)
    idx = find(y_cls==Opt.trainclasses_id(j));
    numpercls_tr(j) = length(idx);
    TrainData = [TrainData; X(idx,:)];
    TrainLabel_K = [TrainLabel_K; Y_kes(idx,:)];
    TrainLabel_c = [TrainLabel_c; y_cls(idx)];
end
numpercls_ts = zeros(length(Opt.testclasses_id),1);
for j=1:length(Opt.testclasses_id)
    idx = find(y_cls==Opt.testclasses_id(j));
    numpercls_ts(j) = length(idx);
    TestData = [TestData; X(idx,:)];
    TestLabel_K = [TestLabel_K; Y_kes(idx,:)];
    TestLabel_c = [TestLabel_c; y_cls(idx)];
end

[Xtr, Xts] = DimNormalization(TrainData, TestData);
TrainData = Xtr;
TestData = Xts;
%---------------------------------------------------------------------------------------
idx_start = 1;
for j=1:length(Opt.trainclasses_id)
    if j>1
        idx_start = sum(numpercls_tr(1:j-1))+1;
    end
    idx_end   = sum(numpercls_tr(1:j));
    TrainDataMu = [TrainDataMu; mean(Xtr(idx_start:idx_end,:))];
end
idx_start = 1;
for j=1:length(Opt.testclasses_id)
    if j>1
        idx_start = sum(numpercls_ts(1:j-1))+1;
    end
    idx_end   = sum(numpercls_ts(1:j));
    TestDataMu = [TestDataMu; mean(Xts(idx_start:idx_end,:))];
end    


VSet_num = length(TrainData(:,1));
VTrainSet_num = VSet_num;
vidx = randperm(VSet_num)';
Data.TrainData = sparse(TrainData(vidx(1:VTrainSet_num),:));
Data.TrainLabel_K = TrainLabel_K(vidx(1:VTrainSet_num),:);
Data.TrainLabel_c = TrainLabel_c(vidx(1:VTrainSet_num));
Data.TrainDataMu = TrainDataMu;
Data.TestData = sparse(TestData);
Data.TestLabel_K = TestLabel_K;
Data.TestLabel_c = TestLabel_c;
Data.TestDataMu = TestDataMu;
Data.trainclasses_id = Opt.trainclasses_id;
Data.testclasses_id = Opt.testclasses_id;
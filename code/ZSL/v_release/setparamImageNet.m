function Opt = setparamImageNet(datasetname)

fprintf(['Set common parameters for ',datasetname,' dataset....\n']);
Opt.path.root = 'Zero-Shot-Recognition-using-Dual-Visual-Semantic-Mapping-Paths/'; % linux
%Opt.path.root = '/CSC4ZSL_v20160828/'; % windows

Opt.inputpath = [Opt.path.root, 'dataset/'];
Opt.outputpath = [Opt.path.root, 'results/'];
Opt.path.feature = [Opt.inputpath, datasetname, '/feature_mat/'];
Opt.path.knowledge = [Opt.inputpath, datasetname, '/knowledge_mat/'];


Opt.dataset = datasetname;
renewSeenUnseen = true;

load([Opt.inputpath, datasetname, '/nperclass.mat']);
Opt.trainsetRate = 0.8;  %for DCP dataset, 80% for train, 20% for test
Opt.nperclass = nperclass;

trainclasses_id = 1:800;
testclasses_id = 801:1000;
Opt.oldtrainclasses_id = trainclasses_id;
Opt.oldtestclasses_id = testclasses_id;
if renewSeenUnseen == true
    idrnd = randperm(length(nperclass))';
    Opt.trainclasses_id = idrnd(1:length(Opt.oldtrainclasses_id));
    Opt.testclasses_id = idrnd(length(Opt.oldtrainclasses_id)+1:end);
else
    Opt.trainclasses_id = Opt.oldtrainclasses_id;
    Opt.testclasses_id = Opt.oldtestclasses_id;
end

Opt.featname = {'GoogleNet1024'};
Opt.vkesname = {'v_GoogleNet1024'};
Opt.KES.name = {'w_cbow5'};
Opt.KES.type = [1];  
end
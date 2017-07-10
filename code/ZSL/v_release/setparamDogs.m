function Opt = setparamDogs(datasetname)

fprintf(['Set common parameters for ',datasetname,' dataset....\n']);
Opt.path.root = '/Zero-Shot-Recognition-using-Dual-Visual-Semantic-Mapping-Paths/';
Opt.inputpath = [Opt.path.root, 'dataset/'];
Opt.outputpath = [Opt.path.root, 'results/'];
Opt.path.feature = [Opt.inputpath, datasetname, '/feature_mat/'];
Opt.path.knowledge = [Opt.inputpath, datasetname, '/knowledge_mat/'];


Opt.dataset = datasetname;

renewSeenUnseen = true;
load([Opt.inputpath, datasetname, '/constants.mat']);
load([Opt.inputpath, datasetname, '/nperclass.mat']);
Opt.trainsetRate = 0.8;
Opt.nperclass = nperclass;
Opt.classes = classes;
Opt.oldtrainclasses_id = trainclasses_id;
Opt.oldtestclasses_id = testclasses_id;
if renewSeenUnseen == true
    idrnd = randperm(length(classes))';
    Opt.trainclasses_id = idrnd(1:length(Opt.oldtrainclasses_id))';
    Opt.testclasses_id = idrnd(length(Opt.oldtrainclasses_id)+1:end)';
else
    Opt.trainclasses_id = Opt.oldtrainclasses_id;
    Opt.testclasses_id = Opt.oldtestclasses_id;
end

Opt.featname = {'vgg', 'goog', 'res', 'vgg+goog', 'vgg+res'};
Opt.vkesname = {'v_vgg', 'v_goog', 'v_res', 'v_vgg+goog', 'v_vgg+res'};
Opt.KES.name = {'w_cbow5', 'w_glove3', 'w_skipgram5'};
Opt.KES.type = [1, 1, 1];
end
clear; close all; clc;
cstart = fix(clock);
fprintf('\n Relational Graph/Manifold Alignment for Zero-Shot Learning, %d-%d-%d %d:%d:%d\n', cstart);
DATASet = {'AwA', 'CUB', 'Dogs', 'ImageNet'};
IterationsSet = [1, 1, 1, 1]*2;
for didx = 1%1:length(DATASet)
    U2T = 1;
    Iterations = IterationsSet(didx);
    fprintf('-----------------------------------------------------------------------------------\n');
    dataset = DATASet{didx};
    Opt  = setparam(dataset);
    for featidx = 1%:length(Opt.featname)
        Opt.featidx = featidx;
        featname = Opt.featname{Opt.featidx};
        for kesidx = 4%1:length(Opt.KES.name)
            fprintf('-----------------------------------------------------------------------------------\n');
            Opt.useVWflag = false;
            Opt.KESidx = kesidx;
            KESname  = Opt.KES.name{Opt.KESidx};
            fprintf('Baseline +++++++++++  Tr, lambda = %.2f, gamma = %.2f, ', log10(Opt.fs.lambda), log10(Opt.fs.gamma));
            [Opt, Data] = generateCSCdata(Opt,2, Opt.KES.type(Opt.KESidx));
            Result = predict_label(Opt, Data);
            constructVKS(Opt, Data, U2T);
            for iter = 1:Iterations
                fprintf('-----------------------------------------------------------------------------------\n');
                fprintf('Iter #%02d ', iter);
                Opt.useVWflag = true;
                Opt.vKESidx = featidx;
                KESname  = Opt.vkesname{Opt.vKESidx};
                [Opt, Data] = generateCSCdata(Opt,2, Opt.KES.type(Opt.KESidx));
                constructVKS(Opt, Data, U2T);
                Result = predict_label(Opt, Data);
            end
            fprintf('-\n-\n');
        end
        fprintf('-----------------------------------------------------------------------------------\n');
        fprintf('-\n-\n-\n-\n-\n');
        cend = fix(clock); 
        thistime = sprintf('%04d-%02d-%02d_h%02dm%02d', cend([1,2,3,4,5]));
    end
cend = fix(clock);
fprintf('-----------------------------------------------------------------------------------\n');
fprintf('Relational  Alignment for Zero-Shot Learning, %d-%d-%d %d:%d:%d\n', cstart);
fprintf('Relational  Alignment for Zero-Shot Learning, %d-%d-%d %d:%d:%d\n', cend);

end
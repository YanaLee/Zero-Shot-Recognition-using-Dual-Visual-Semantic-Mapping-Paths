function Opt = setparam(datasetname)
switch datasetname
    case 'AwA'
        Opt  = setparamAwA(datasetname);
		Opt.kTop = 200;       
        lparaspace = 0.2; 
        gparaspace = 3; 
            
    case 'CUB'
        Opt  = setparamCUB(datasetname);
		Opt.kTop = 200;        
        lparaspace = 0.5;
        gparaspace = 1.4;         

    case 'Dogs'
        Opt  = setparamDogs(datasetname);
		Opt.kTop = 120;        
        lparaspace = 0.1;
        gparaspace = 2.1;
    case 'ImageNet'
        Opt  = setparamImageNet(datasetname);
		Opt.kTop = 100;        
        lparaspace = 0.6;
        gparaspace = 3.0;                
end

Opt.CSC.gamma = 0.0001; % parameter for ridge regression
Opt.CSC.lambda = 0.1;   % sparse degree for LeastR function
Opt.CSC.L = 150;        % the max. number of coefficients for each signal for OMP function

Opt.kTop1 = 100;        % parameter for constructing K transductively
Opt.rTop = 1.0;         % parameter for constructing K transductively
Opt.fs.lambda = 10^lparaspace; % optimization parameter for learning fs
Opt.fs.gamma = 10^gparaspace;  % optimization parameter for learning fs
end
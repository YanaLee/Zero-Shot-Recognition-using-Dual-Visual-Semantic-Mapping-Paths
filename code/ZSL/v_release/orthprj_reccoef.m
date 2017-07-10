%% compute the reconstruction coefficients of orthogonal projection vector
function RCoef = orthprj_reccoef(Opt, Ms, Mt, prjmodel)
rc = zeros(size(Ms,2), size(Mt,2));
switch prjmodel
    case 1 % linear regression for reconstruction
        %RCoef = inv(Ms'*Ms)*Ms'*Mt;
        rc = (Ms'*Ms)\(Ms'*Mt);
    case 2 % ridge regression for reconstruction
        rc = (Ms'*Ms + Opt.CSC.gamma*eye(size(Ms,2)))\(Ms'*Mt);
    case 3 % sparse coding for reconstruction
        for k = 1 : size(Mt,2)
            [rc(:, k)] = LeastR(Ms, Mt(:, k), Opt.CSC.lambda);
            %LL = min(size(Ms,2), Opt.CSC.L);
            %[rc(:, k)] = OMP(Ms, Mt(:, k), LL);
        end
end
RCoef = rc;
end
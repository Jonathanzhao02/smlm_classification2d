% all2all_class   performs all2all registration for a given set of particles
%
% SYNOPSIS:
%   all2all_class(Particles, scale)
%
% INPUT
%   Particles
%       Cell arrays of particles with localization in the point field and
%       squared uncertainties in the sigma field.
%   scale
%       scale parameter for gmm registration
%
% OUTPUT
%   The function is similar to its equivalent 'all2all.m, but returns the
%   output instead of saving it. Each element of the all2all matrix includes registration
%   parameters (result.parameter), cost function value (result.val) and the
%   indicator pair (result.id) which stores (row, column) indices of all2all 
%   matrix
%
% (C) Copyright 2017               Quantitative Imaging Group
%     All rights reserved          Faculty of Applied Physics
%                                  Delft University of Technology
%                                  Lorentzweg 1
%                                  2628 CJ Delft
%                                  The Netherlands
%
% Teun Huijben, Dec 2020

function Matrix = all2all_class(Particles, scale, nAngles)

    % setup pyramid, determine the pyramid height or the number of layers
    N = numel(Particles);
    
    disp('all2all registration started !');
    disp(['There are ' num2str(N-1) ' rows !']);
    for idxM = 1:N-1
        tic
        disp(['row ' num2str(idxM) ' started!']);
        parfor idxS = idxM+1:N
            M = Particles{idxM};
            S = Particles{idxS};
            % perform pairwise registration for each element of all2all
            % matrix
            
            [param, ~, ~, ~, val] = pairFitting(M, S, scale, nAngles);

            % registration parameters, cost function value and idicators
            % are stored in the result structure
            Matrix(idxM,idxS).parameter = param;
            Matrix(idxM,idxS).val = val;
            Matrix(idxM,idxS).id = [idxM;idxS];

        end
        
        clearvars result;
        a = toc;
        disp(['row ' num2str(idxM) ' done in ' num2str(a) ' seconds']);

    end

    disp('all2all registration done !');
    fprintf('\n\n');
    
end








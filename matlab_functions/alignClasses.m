%alignClasses   Align the classes with each other
%
%   SYNOPSIS:
%       [superParticle_class] = alignClasses(subParticles, clusters, super, scale)
%
%   Input: 
%       subParticles
%           Cell array of particles of size 1xN, with localization in the point field and 
%           uncertainties in the sigma field.
%       clusters
%           cell array, where each entry contains a vector with
%           particle IDs belonging to that specific class
%       super
%           Cell array containing of size (C,I), for C classes and
%           I bootstrap iterations. (only the last iteration will be used for
%           each class
%       scale
%           parameter for registration
%
%   Output:
%       superParticle_class: the resulting fused particle
%
%
% (C) Copyright 2017                    QI Group
%     All rights reserved               Faculty of Applied Physics
%                                       Delft University of Technology
%                                       Lorentzweg 1
%                                       2628 CJ Delft
%                                       The Netherlands
%
% Teun Huijben, Dec 2020.

function [superParticle_class] = alignClasses(subParticles, clusters, super, scale, nAngles)

K = length(clusters); 

for i = 1:K
    members = clusters{i}';
    sigmas = [];
    for m = members
        sigmas = [sigmas; subParticles{m}.sigma];
    end
    subParticles_clustered{i}.points = super{i}{end}; 
    subParticles_clustered{i}.sigma = sigmas; 
end

len = cellfun(@(v) size(v.points,1), subParticles_clustered);
mat = zeros(K,K); 
mat_norm = zeros(K,K); 

subParticles_clustered_subsamp = subParticles_clustered;
for i = 1:K
     ind = randsample(1:len(i),min(len)); 
     subParticles_clustered_subsamp{i}.points = subParticles_clustered_subsamp{i}.points(ind,:); 
     subParticles_clustered_subsamp{i}.sigma = subParticles_clustered_subsamp{i}.sigma(ind); 
end

all2allMatrix = all2all_class(subParticles_clustered_subsamp,scale,nAngles); 
[initAlignedParticles_class, M1_class] = outlier_removal_class(subParticles_clustered,all2allMatrix);
iter = 3; 
[~,~,superParticle_class] = one2all_class(initAlignedParticles_class,iter,M1_class,scale,nAngles); 

end


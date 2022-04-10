% This code loads a dataset from the '/data' directory 
% which should be a cell array of structures called 
% subParticles with localizations in the 'point' field 
% [x,y] and SQUARED uncertainties in 'sigma' field: 
% subParticles{1}.points  -> localization data (x,y) in camera pixel units
% subParticles{1}.sigma   -> localization uncertainties (sigma) in SQUARED pixel units
% 
% The following code will load the data from file and 
% then performs the 4 (or 5) steps of the classification 
% algorithm. Different example datasets are provided, 
% both experimental as simulated data. You only need to 
% provide the 'dataset' name, and the values for K (and 
% optionally C)
%
% The code makes use of the parallel computing toolbox 
% to distribute the load over different workers. 
% 
% (C) Copyright 2017               Quantitative Imaging Group
%     All rights reserved          Faculty of Applied Physics
%                                  Delft University of Technology
%                                  Lorentzweg 1
%                                  2628 CJ Delft
%                                  The Netherlands
%
% Teun Huijben, 2020
%%  
close all
clear all
clc

% path(pathdef)
% add the required directory to path
addpath(genpath('datafusion2d'))
addpath(genpath('matlab_functions'))

%% LOAD DATASET
% -- select data set ---
% dataset = 'ASU_2red_300';
% dataset = 'ASU_3red_300';
dataset = 'NSF_335';
% dataset = '200x_simulated_TUD_flame';           %100 with flame, 100 without flame (80% DoL)
% dataset = '200x_simulated_TUD_mirror';          %10 mirrored, 190 normal (80% DoL)
% dataset = '456_experimental_TUD_mirror';     %experimental dataset of which a few (~2%) are mirrored

% -- choose number of particles --
% N = 200;     %length(subparticles)

% -- set number of classes --
K = 4;          %set to 2 for the simulated TUD_flame dataset, this will give the correct classes
                    %set to 4 for the other two datasets, and continue with STEP 5 using C=2        

% -- choose number of final classes (C<K) --
C = 3; 

% -- set width of visualization --
width = 1.0;
furtherClustering = false;
scale = 0.03;
nAngles = 12;

load(['data/' dataset '/subParticles.mat'])
N = length(subParticles);

outdir = ['output/' dataset];
if ~exist(outdir,'dir')
    mkdir(outdir);
else
    disp('Warning: outdir already exists')
end    

save([outdir '/subParticles'], 'subParticles');


%Assuming the all2all registration is already performed, we can load the
%dissimilarity matrix into our newly generated output-folder:

% copy_from = ['data/' dataset '/all2all_matrix/'];
% copy_to = [outdir '/all2all_matrix/'];

% copyfile(copy_from, copy_to)

            % 0.01 for experimental TUD (in camera pixel units)
            % 0.03 for simulated TUD
            % 0.1 for NPC
            % 0.03 or Digits data
            % 0.15 for Letters data                
            % Look at Online Methods for the description

%% STEP 2: Multi-dimensional scaling
disp("Starting multi-dimensional scaling!");

% load the similarity matrix and normalize with respect to number of localizations;

disp("Loading similarity matrix!");

[SimMatrix, SimMatrixNorm] = MakeMatrix(outdir,subParticles,N);  

% dissimilarity matrix
disp("Calculating full similarity matrix!");

D = SimMatrixNorm+SimMatrixNorm';                   % convert upper-triangular matrix to full similarity matrix
D = max(D(:))-D;                                                     % convert to dissimilarity
D = D - diag(diag(D));                                             % set diagonal to 0
save([outdir '/similarity_matrix'], 'D');

disp("Performing MDS!");

tstart = tic;

try
    mds = mdscale(D,30,'Criterion','metricstress');     % perform multi-dimensional scaling
catch E
    disp('Ran into error');
    disp(E);
    quit
end

toc(tstart)

disp("Saving figure!");

% show first three dimensions of MDS 
f = figure('visible', 'off'); scatter3(mds(:,1),mds(:,2),mds(:,3),'o','filled'), title 'Multidimensional Scaling'
saveas(f, [outdir '/MDS_3D.fig'], 'fig')
close(f)

disp("Finished multi-dimensional scaling!");

%% STEP3: k-means clustering
disp("Starting k-means clustering!");

tstart = tic;

clus = kmeans(mds,K,'replicates',1000);
[clus, K] = recluster(clus,K,mds);

toc(tstart)

clear clusters
for i = 1:K
    clusters{i} = find(clus==i);
end

save([outdir '/clusters'], 'clusters');

disp("Saving figure!");

f = figure('visible', 'off'); scatter3(mds(:,1),mds(:,2),mds(:,3),[],clus,'o','filled'), title 'Clustering result'
saveas(f, [outdir '/MDS_3D_clustered.fig'], 'fig')
close(f)

disp("Finished k-means clustering!");

%% STEP 4: Reconstruction per cluster
disp("Starting reconstruction per cluster!");

iters = 2;      %number of bootstraps

tstart = tic;

[~,classes] = reconstructPerClassFunction(subParticles,clusters,outdir,scale,iters,nAngles);

toc(tstart)

save([outdir '/classes'], 'classes');

%% Visualize results
close all

%random particle
ran = randi(N); 
f = figure('visible', 'off');
visualizeCloud2D(subParticles{ran}.points,200,width,0,'example particle',f);
saveas(f, [outdir '/rand_particle.fig'], 'fig')
close(f)

%fusion of all particles without classification
%f = figure('visible', 'off');
%visualizeCloud2D(superParticle{end},200,width,0,'superParticle',f);

% reconstructed clusters
for i = 1:length(classes)
    str = ['class ' num2str(i) ' (' num2str(length(clusters{i})) ' particles)' ];
    f = figure('visible', 'off');
    visualizeCloud2D(classes{i}{end},200,width,0,str,f);
    saveas(f, [outdir '/class_' num2str(i) '.fig'], 'fig')
    close(f)
end

disp("Finished reconstruction per cluster!");

%% (optional) STEP 5: further clustering - Eigen image method (C<K)
if furtherClustering
    disp("Starting further clustering!");

    try
        classes_aligned = alignClasses(subParticles, clusters, classes, scale, nAngles); 
        classes_merged = eigenApproach(classes_aligned,C,width);
    catch E
        disp('Ran into error');
        disp(E);
        exit
    end

    for i = 1:C
        f = figure('visible', 'off');
        visualizeCloud2D(classes_merged{i},200,width,0,['class: ' num2str(i)],f);
        saveas(f, [outdir '/class_merged_' num2str(i) '.fig'], 'fig')
        close(f)
    end

    disp("Finished further clustering!");
end

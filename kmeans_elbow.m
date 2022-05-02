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

% -- set max number of classes --
K = 30;          %set to 2 for the simulated TUD_flame dataset, this will give the correct classes
                    %set to 4 for the other two datasets, and continue with STEP 5 using C=2        

load(['data/' dataset '/subParticles.mat'])
N = length(subParticles);

outdir = ['output/' dataset];
if ~exist(outdir,'dir')
    mkdir(outdir);
else
    disp('Warning: outdir already exists')
end    

save([outdir '/subParticles'], 'subParticles');

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

disp("Performing MDS!");

try
    mds = mdscale(D,30,'Criterion','metricstress');     % perform multi-dimensional scaling
catch E
    disp('Ran into error');
    disp(E);
    quit
end

disp("Finished multi-dimensional scaling!");

%% STEP3: k-means clustering
disp("Starting k-means clustering!");

for i = 1:K
    disp(['Clustering on ' num2str(i)]);
    [idx, ~, sumd] = kmeans(mds,i,'replicates',1000);
    disp(sumd');
    inertias(i) = sum(sumd);
end

disp("Finished k-means clustering!");

plot(1:K, inertias);

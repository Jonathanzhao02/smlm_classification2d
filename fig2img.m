% This code loads a dataset from the '/output' directory 
% 
% The following code will load the .fig files from the directory
% and convert them to .png files.
%%  
close all
clear all
clc

% add the required directory to path
addpath(genpath('datafusion2d'))
addpath(genpath('matlab_functions'))

%% LOAD DATASET
% -- select data set ---
dataset = 'NSF_120';
% dataset = 'Cropped_Bima_NSF-1nM_R1-5nM_50ms_TIRF_50Kframes_70mW_locs_render_filter_picked';
% dataset = '200x_simulated_TUD_flame';           %100 with flame, 100 without flame (80% DoL)
% dataset = '200x_simulated_TUD_mirror';          %10 mirrored, 190 normal (80% DoL)
% dataset = '456_experimental_TUD_mirror';     %experimental dataset of which a few (~2%) are mirrored

outdir = ['output/' dataset];
if ~exist(outdir,'dir')
    disp('Error: outdir does not exist');
    quit;
end

%% STEP 2: Multi-dimensional scaling
disp("Saving MDS image!");

% show first three dimensions of MDS
try
    f = openfig([outdir '/MDS_3D.fig'], 'invisible');
    saveas(f, [outdir '/MDS_3D.png'], 'png')
    close(f)
catch E
    disp('Could not save MDS image');
    disp(E);
end

%% STEP3: k-means clustering
% -- set number of classes --
K = 4;          %set to 2 for the simulated TUD_flame dataset, this will give the correct classes
                    %set to 4 for the other two datasets, and continue with STEP 5 using C=2        

disp("Saving k-means clustering image!");

try
    f = openfig([outdir '/MDS_3D_clustered.fig'], 'invisible');
    saveas(f, [outdir '/MDS_3D_clustered.png'], 'png')
    close(f)
catch E
    disp('Could not save k-means clustering image');
    disp(E);
end

%% Visualize results
close all

%random particle
disp('Saving random particle image!');

try
    f = openfig([outdir '/rand_particle.fig'], 'invisible');
    saveas(f, [outdir '/rand_particle.png'], 'png')
    close(f)
catch E
    disp('Could not save random particle image');
    disp(E);
end

%fusion of all particles without classification
% visualizeCloud2D(superParticle{end},200,width,0,'superParticle');

% reconstructed clusters
disp('Saving reconstructed cluster images!');
for i = 1:K
    try
        f = openfig([outdir '/class_' num2str(i) '.fig'], 'invisible');
        saveas(f, [outdir '/class_' num2str(i) '.png'], 'png')
        close(f)
    catch E
        disp(['Errored on cluster ' num2str(i)]);
        disp(E);
    end
end

%% (optional) STEP 5: further clustering - Eigen image method (C<K)
% -- choose number of final classes (C<K) --
C = 3;

disp('Saving further clustering images!');
for i = 1:C
    try
        f = openfig([outdir '/class_merged_' num2str(i) '.fig'], 'invisible');
        saveas(f, [outdir '/class_merged_' num2str(i) '.png'], 'png')
        close(f)
    catch E
        disp(['Errored on cluster ' num2str(i)]);
        disp(E);
    end
end

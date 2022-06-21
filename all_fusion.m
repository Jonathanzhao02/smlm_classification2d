for dset={'A_2red_211_labeled','A_3red_206_labeled','S_2red_206_labeled','S_3red_210_labeled','U_2red_206_labeled','U_3red_207_labeled'}
    close all
    clearvars -except dset
    clc

    dataset = dset{1};
    
    % path(pathdef)
    % add the required directory to path
    addpath(genpath('datafusion2d'))
    addpath(genpath('matlab_functions'))
    
    %% LOAD DATASET
    % -- select data set ---
    % dataset = 'binding32nt_20nm';
    % dataset = 'ASU_2red_30_labeled';
    % dataset = 'ASU_2red_300';
    % dataset = 'ASU_3red_300';
    % dataset = 'NSF_335';
    % dataset = '200x_simulated_TUD_flame';           %100 with flame, 100 without flame (80% DoL)
    % dataset = '200x_simulated_TUD_mirror';          %10 mirrored, 190 normal (80% DoL)
    % dataset = '456_experimental_TUD_mirror';     %experimental dataset of which a few (~2%) are mirrored
    
    % -- set width of visualization --
    width = 1.0;
    scale = 0.03;
    nAngles = 12;
    
    load(['data/' dataset '/subParticles.mat'])
    N = length(subParticles);
    
    outdir = ['old/output/' dataset];
    if ~exist(outdir,'dir')
        mkdir(outdir);
    else
        disp('Warning: outdir already exists')
    end
    
    %optional fusion of all particles (not necessary for classification)
    
    disp("Starting optional fusion!");
    
    [initAlignedParticles, M1] = outlier_removal(subParticles, [outdir '/all2all_matrix/'], outdir);        %Lie-algebraic averaging
    iters = 3;                                                                                                                               %number of bootstrap iteration
    [superParticle, ~] = one2all(initAlignedParticles, iters, M1, outdir,scale, nAngles);                                      %bootstrapping
    
    disp("Finished optional fusion!");
    
    %fusion of all particles without classification
    f = figure('visible', 'off');
    visualizeCloud2D(superParticle{end},200,width,0,'',f);
    saveas(f, [outdir '/superParticle.fig'], 'fig')
    close(f)
end
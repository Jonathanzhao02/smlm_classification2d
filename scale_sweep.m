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

load(['data/' dataset '/subParticles.mat'])
N = length(subParticles);

%% Scale sweep
clear val

nAngles = 12;
scales = linspace(0.001,0.5,60); %dependent on whether localisations are in pixel/nm units

for t = 1:20
    order = randperm(N);  %shuffle all particles
    idxM = order(1);
    idxS = order(2);  

    M = subParticles{idxM};
    S = subParticles{idxS};
    
    M.points = double(M.points);
    S.points = double(S.points);
    M.sigma = double(M.sigma);
    S.sigma = double(S.sigma); 

    parfor i = 1:length(scales)
        [t,i]
        [~, ~, ~, ~, val(t,i)] = pairFitting(M, S, scales(i),nAngles);
    end
end

val_norm = val./repmat(max(val,[],2),1,length(scales)); 

figure()
for i = 1:size(val,1)
    plot(scales,val_norm(i,:))
    hold on
end
hold on
plot(scales,mean(val_norm,1),'k','LineWidth',5)
xlabel('scale (pixels)')
ylabel('costValue (norm. per line)')
title('Scale Sweep')
grid; grid minor;

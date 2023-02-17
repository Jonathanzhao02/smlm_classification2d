% The following code will load a result from the
% clustering and show the breakdowns of each
% individual cluster, given the data is labeled.
%%  
close all
clear all
clc

% add the required directory to path
addpath(genpath('datafusion2d'))
addpath(genpath('matlab_functions'))

% Main code
%%
possible = ['A', 'S', 'U'];

picks = open('output/ASU_2red_584_labeled_cropped/final.mat').picks;
p = cell2mat(picks);

breakdowns = containers.Map();

for i=unique([p.cluster])
    breakdowns(num2str(i)) = containers.Map();
    b = breakdowns(num2str(i));

    for key=possible
        b(key) = 0;
    end
end

for i=1:numel(picks)
    pick = picks{i};
    b = breakdowns(num2str(pick.cluster));

    for key=possible
        if startsWith(pick.group, key)
            b(key) = b(key) + 1;
        end
    end
end

for i=unique([p.cluster])
    fprintf('Cluster %d\n', i);
    b = breakdowns(num2str(i));

    for key=possible
        fprintf('%s: %d\n', key, b(key));
    end
end

clear all
clc

addpath(genpath('datafusion2d'))
addpath(genpath('matlab_functions'))

width = 1.0;
dataset = 'ASU_2red_584_labeled_cropped';
outdir = ['output/' dataset];

clusters = load([outdir '/classes.mat']).classes;

for i = 1:length(clusters)
    str = '';
    f = figure('visible', 'off');
    visualizeCloud2D(clusters{i}{end},200,width,0,str,f);
    saveas(f, [outdir '/class_' num2str(i) '.fig'], 'fig')
    close(f)
end

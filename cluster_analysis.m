% Note: only NSF dataset has individual tags
dataset = 'NSF_335';
% dataset = 'ASU_2red_300';
% dataset = 'ASU_3red_300';
K = 4;
tags = ['N' 'S' 'F'];
% tags = ['A', 'S', 'U'];

outdir = ['output/' dataset];
load([outdir '/clusters.mat'])
load([outdir '/subParticles.mat'])
N = length(subParticles);

for i=1:K
    classParticles = [ subParticles{clusters{i}} ];
    classTags = { classParticles.group };
    disp(['Sizeof cluster ' num2str(i) ': ' num2str(numel(classParticles))]);
    
    for j=1:numel(tags)
        tagCount = 0;
        tag = tags(j);
        
        for classTag=classTags
            if startsWith(classTag, tag)
                tagCount = tagCount + 1;
            end
        end
        
        disp(['Clusters with tag ' tag ': ' num2str(tagCount)]);
    end
    
    disp(newline);
end

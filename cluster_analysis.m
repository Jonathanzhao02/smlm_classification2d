dataset = 'NSF_120';
K = 3;
tags = ['N' 'S' 'F'];

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

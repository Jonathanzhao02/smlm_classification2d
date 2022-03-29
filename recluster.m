% Merges clusters of size 1 with nearby clusters

function [clusters,K] = recluster(clusters,K,mds)
mergers = [];

for i=1:K
    clusterIds{i} = find(clusters==i);

    if numel(clusterIds{i}) == 1
        mergers(end + 1) = i;
    end
end

order = randperm(numel(mergers));
removed = [];

for i=order
    c = mergers(i);
    idx = clusterIds{c};

    if numel(idx) == 1
        for j=1:K
            if j ~= c
                dists(j) = pdist2(mds(idx,:),mean(mds(clusterIds{c},:),1),'euclidean');
            else
                dists(j) = inf;
            end
        end
    
        [~,I] = min(dists);
        clusterIds{I}(end + 1) = idx;
        clusters(idx) = I;
        removed(end + 1) = c;
    end
end

removed = sort(removed, 'desc');

for i=removed
    ind = clusters > i;
    clusters(ind) = clusters(ind) - 1;
end

K = K - numel(removed);
fprintf("Removed %d clusters\n", numel(removed));

end

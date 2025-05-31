function patches = ZCAwhitenData_s2(patches)
numpatches=size(patches,2);
 epsilon=0.1;
sigma = patches * patches' / numpatches;
[u, s, v] = svd(sigma);
ZCAWhite_s2 = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';%
patches = ZCAWhite_s2 * patches;
save ZCAWhite_s2.mat ZCAWhite_s2;
end

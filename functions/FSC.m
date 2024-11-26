function y = FSC(X,c,numAnchor,numNearestAnchor)

X =(double(X));%n*d
num = size(X,1);
X = X-repmat(mean(X),[num,1]);
[~, locAnchor] = kmeans(X, numAnchor);
Z = ConstructA_NP(X',locAnchor',numNearestAnchor);
Z2 = full(Z);
a2 = sum(Z2,1);
index2 = find(a2~=0);
a2(index2) = 1./a2(index2);
D2a = diag(a2);
n = size(Z2,1);
A1 = Z2*D2a;
SS2 = A1'*A1;
[V, ev0, ~]=eig1(SS2,c);
V = V(:,1:c);
U=(A1*V)./(ones(n,1)*sqrt(ev0(1:c)'));
y = kmeans(U,c);
end

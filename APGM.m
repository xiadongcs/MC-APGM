function [Z,changed,obj,Wv] = APGM(X,groundtruth,graph,k_nn_num,p,th,v_n)
classnum = max(groundtruth);
viewnum = length(X);   
n = length(groundtruth); 
[num,~] = size(X{1}); 
Fv = cell(1,viewnum); 
Av_rep = zeros(num);  
X = X';

for i = 1 :viewnum
    for  j = 1:n
         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) );
    end
end

options.NeighborMode = 'KNN';
options.k = k_nn_num; 
options.WeightMode = 'HeatKernel';
for v = 1:viewnum
    Xv = X{v};
    if graph == 1
        Av = constructW(Xv,options); 
    else
        Av = constructW_PKN(Xv',k_nn_num);
    end
    Av_rep = Av + Av_rep;    
    Lv = Ls(Av);
    temp = eig1(full(Lv),classnum+1,0);
    if th == 1
       Fv{v} = temp(:,2:classnum+1);
       Fv{v} = Fv{v}./repmat(sqrt(sum(Fv{v}.^2,2)),1,classnum); 
    elseif th == 2  
       Fv{v} = temp(:,2:classnum+1);
    elseif th == 3
       Fv{v} = temp(:,2:classnum+1);
       Fv{v} = Fv{v}./repmat(sqrt(sum(Fv{v}.^2,2)),1,classnum); 
       Fv{v} = orth(Fv{v});
    end
end

Pv = Fv;
F = Fv{v_n};
Z = diag(diag(F*F'))^(-0.5)*F;
for i = 1:size(Z,1)
    [~,mix] = max(Z(i,:));
    Z(i,:) = 0;
    Z(i,mix) = 1;
end

NITER = 20;
changed = zeros(NITER,1);

obj = [];
for iter = 1:NITER

M = Z*(Z'*Z)^(-0.5);

for v = 1:viewnum
    [tmp_u, ~, tmp_v] = svd(Pv{v}'*M);
    Qv{v} = tmp_u * tmp_v';
end 

for v = 1:viewnum
    Wv(v) = 0.5*p*norm(M - Pv{v}*Qv{v}, 'fro')^(p-2); 
end  
    
tem = 0;
for v = 1:viewnum
    tem = tem + norm(M - Pv{v}*Qv{v}, 'fro')^p; 
end
obj = [obj; tem];
if iter>=2 && obj(iter-1)-obj(iter)<10^-4
   break;
end

G = zeros(num,classnum);
for v = 1:viewnum
    G = G + Wv(v)*Pv{v}*Qv{v};
end
[WPQ,g] = max(G,[],2);
Z = TransformL(g,classnum);
[~,ind] = sort(WPQ);
zg = diag(Z'*G);
zz = diag(Z'*Z);
    
    for it = 1:10
        converged = 0;
        for i = 1:num
            N1 = zg' + G(ind(i),:).*(1-Z(ind(i),:));
            DE1 = zz' + (1-Z(ind(i),:));
            
            N2 = zg' - G(ind(i),:).*Z(ind(i),:);
            DE2 = zz' - Z(ind(i),:);
            
            [~,id1] = max(N1./sqrt(DE1)-N2./sqrt(DE2));
            id0 = find(Z(ind(i),:)==1);
            
            if id1 ~= id0
                Z(ind(i),:) = 0;
                Z(ind(i),id1) = 1;
                zg(id0) = zg(id0) - G(ind(i),id0);
                zg(id1) = zg(id1) + G(ind(i),id1);
                zz(id0) = zz(id0) - 1;
                zz(id1) = zz(id1) + 1;
                converged = converged + 1;
            end
        end
        
        if converged == 0
            break;
        end
    end

    changed(iter) = it;

end

[~, Z] = max(Z,[],2);

end

function L0 = Ls(A)
    S0 = A;
    S10 = (S0+S0')/2;
    D10 = diag(sum(S10));
    L0 = D10 - S10;
end
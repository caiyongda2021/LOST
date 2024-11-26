%LOST: Efficient and Accurate Ensemble Clustering
%Yongda Cai, Dingming Wu, Tsz Nam Chan, Xudong Sun, Joshua Zhexue Huang

clear;clc
close all

addpath('datasets')
addpath('functions')

load SF2M_geo_distributed_data_centers.mat
warning('off')

ticStart = tic;
Y_true = [Y_DC1;Y_DC2;Y_DC3;Y_DC4];
numRSP = 20;
numRSP_sample = 5;
delta = 0.8;
repeatTime = 20;

for iter_exp = 1:repeatTime
    % Generate random samples of big data
    % random sample partition on DC1
    [num,~] = size(X_DC1);
    indexanchor_DC1 = randi([1, num], [1, num]);
    interval = floor(num/numRSP);
    for iter = 1:numRSP
            if iter~=numRSP
                X_DC1_sample(:,:,iter)= X_DC1(indexanchor_DC1((iter-1)*interval+1:iter*interval),:);
            else
                X_DC1_sample(:,:,iter)= X_DC1(indexanchor_DC1(end-interval+1:end),:);
            end 
    end
    % random sample partition on DC2
    [num,~] = size(X_DC2);
    indexanchor_DC2 = randi([1, num], [1, num]);
    interval = floor(num/numRSP);
    for iter = 1:numRSP
            if iter~=numRSP
                X_DC2_sample(:,:,iter)= X_DC2(indexanchor_DC2((iter-1)*interval+1:iter*interval),:);
            else
                X_DC2_sample(:,:,iter)= X_DC2(indexanchor_DC2(end-interval+1:end),:);
            end 
    end
    % random sample partition on DC3
    [num,~] = size(X_DC3);
    indexanchor_DC3 = randi([1, num], [1, num]);
    interval = floor(num/numRSP);
    for iter = 1:numRSP
            if iter~=numRSP
                X_DC3_sample(:,:,iter)= X_DC3(indexanchor_DC3((iter-1)*interval+1:iter*interval),:);
            else
                X_DC3_sample(:,:,iter)= X_DC3(indexanchor_DC3(end-interval+1:end),:);
            end 
    end
    % random sample partition on DC4
    [num,~] = size(X_DC4);
    indexanchor_DC4 = randi([1, num], [1, num]);
    interval = floor(num/numRSP);
    for iter = 1:numRSP
            if iter~=numRSP
                X_DC4_sample(:,:,iter)= X_DC4(indexanchor_DC4((iter-1)*interval+1:iter*interval),:);
            else
                X_DC4_sample(:,:,iter)= X_DC4(indexanchor_DC4(end-interval+1:end),:);
            end 
    end
    % Select local random samples and combine them as a global random
    % sample
    sample_index = randi(numRSP,1,4);
    X_sample=[X_DC1_sample(:,:,sample_index(1));X_DC2_sample(:,:,sample_index(2));X_DC3_sample(:,:,sample_index(3));X_DC4_sample(:,:,sample_index(4))];
    clear X_DC1_sample X_DC2_sample X_DC3_sample X_DC4_sample
    % random sample partition on global random sample
    [num,dim] = size(X_sample);
    indexanchor_sample = randi([1, num], [1, num]);
    interval = floor(num/numRSP_sample);
    for iter = 1:numRSP_sample
            if iter~=numRSP_sample
                X_sample_RSP(:,:,iter)= X_sample(indexanchor_sample((iter-1)*interval+1:iter*interval),:);
            else
                X_sample_RSP(:,:,iter)= X_sample(indexanchor_sample(end-interval+1:end),:);
            end 
    end
    %Generate base clustering results of global random sample
    k = numel(unique(Y_true));  
    E = 20; %Number of base clusterings
    t0 = [];
    t = 15; %Ensemble size (Second layer)
    upK = k*16;
    lowK = k*15;
    Ks = randsample(upK-lowK+1,E,true)+lowK-1; %Number of clusters in first layer
    Ks_new = k*3; %Number of clusters in second layer
    C_model = [];
    for iter = 1:E
        [y_temp,C_temp]=kmeans(X_sample_RSP(:,:,randperm(numRSP_sample,1)),Ks(iter),'MaxIter',30);
        C = C_temp;
        C_model = [C_model;C];
        Y_C = 1:Ks(iter);
        savefile = ['tempdata/C',num2str(iter),'.mat'];
        save(savefile,'C','Y_C')      
    end
    %Generate B matrix
    Y_all = [];
    for iter = 1:E
        B = [];
        loadfile = ['tempdata/C',num2str(iter),'.mat'];
        load(loadfile,'C','Y_C')
        Mdl = fitcknn(C,Y_C);
        Y_sample = predict(Mdl,X_sample);
        Y_all = [Y_all,Y_sample];
        B = generateBinaryMatrix(Y_sample,Ks(iter));
        savefile = ['tempdata/B',num2str(iter),'.mat'];
        save(savefile,'B')
        clear B
    end
    %calculate t0
    for iter_t0 = 5:15
        t0 = iter_t0;
        Y_sample1 = calculateEnsemble(E,t0,Ks_new);
        Y_sample2 = calculateEnsemble(E,t0,Ks_new);
        result_all = ClusteringMeasure_new(Y_sample1,Y_sample2);
        NMI = result_all(2);
        F1socre = result_all(4);
        if NMI>delta
            break;
        end
        if isnan(F1socre)==true
            break;
        end
    end
    %ensemble clustering
    maxKmIters=50;
    Y_all_new = [];
    B4 = [];   
    for iter2 = 1:t    
        [Y_meta,KK] = calculateEnsembleFinal(Y_all,E,t0,Ks_new);
        B4 = [B4,generateBinaryMatrix(Y_meta,Ks_new)];
    end
    B4_old = B4;
    % BSGP
    B4 = B4./sqrt(t)*diag(1./(sqrt(sum(B4)+1e-6))); 
    B4TB4 = B4'*B4;
    [V1, D1] = eigs(B4TB4,k);
    V1 = bsxfun( @rdivide, V1, sqrt(sum(V1.*V1,2)) + 1e-10 );
    Y_cluster_final = kmeans(V1,k,'MaxIter',maxKmIters);
    %PVE*
    U = generateBinaryU(Y_cluster_final,k);
    Y_sample_temp = B4_old*U';
    [~,Y_sample_ensemble_clustering] = max(Y_sample_temp,[],2);
    %Generating clustering result of big data
    Mdl_sample = fitcknn(X_sample,Y_sample_ensemble_clustering);
    Y_C_all = predict(Mdl_sample,C_model);   
    Mdl_C_all = fitcknn(C_model,Y_C_all);
    Y_DC1_clustering = predict(Mdl_C_all,X_DC1);
    Y_DC2_clustering = predict(Mdl_C_all,X_DC2);
    Y_DC3_clustering = predict(Mdl_C_all,X_DC3);
    Y_DC4_clustering = predict(Mdl_C_all,X_DC4);

    Y_clustering = [Y_DC1_clustering;Y_DC2_clustering;Y_DC3_clustering;Y_DC4_clustering];
    clear B3
    result(iter_exp,:) = ClusteringMeasure_new(Y_clustering,Y_true) 
    clear Y_DC1_clustering Y_DC2_clustering Y_DC3_clustering Y_DC4_clustering Y_clustering
end

runningTime = toc(ticStart);
ave = mean(result)
time = runningTime/20
save result_temp.mat result time

function Y_sample = calculateEnsemble(E,F,Ke_new)
    KK = randperm(E);
    maxKmIters=50;
    B3 = [];
    for iter1 = 1:F
        loadfile = ['tempdata/B',num2str(KK(iter1)),'.mat'];
        load(loadfile,'B')
        B3 = [B3,B];
    end
    k = Ke_new;   
    B3_old = B3;
    B3 = B3./sqrt(F)*diag(1./sqrt(sum(B3)));  
    B3TB3 = B3'*B3;
    [V1, ~] = eigs(B3TB3,k);
    V1 = bsxfun( @rdivide, V1, sqrt(sum(V1.*V1,2)) + 1e-10 );
    Y_cluster = kmeans(V1,k,'MaxIter',maxKmIters);
    U = generateBinaryU(Y_cluster,k);
    Y_sample_temp = B3_old*U';
    [~,Y_sample] = max(Y_sample_temp,[],2);
end

function [Y_sample,KK] = calculateEnsembleFinal(Y_all_input,E,F,Ke_new)
    KK = randperm(E);
    B3 = [];
    for iter1 = 1:F
        loadfile = ['tempdata/B',num2str(KK(iter1)),'.mat'];
        load(loadfile,'B')
        B3 = [B3,B];
    end
    Y_all = Y_all_input(:,KK(1:F));
    k = Ke_new;
    maxKmIters=50;
    B3_old = B3;
    B3 = B3./sqrt(F)*diag(1./sqrt(sum(B3)));  
    B3TB3 = B3'*B3;
    [V1, ~] = eigs(B3TB3,k);
    V1 = bsxfun( @rdivide, V1, sqrt(sum(V1.*V1,2)) + 1e-10 );
    Y_cluster = kmeans(V1,k,'MaxIter',maxKmIters);
    U = generateBinaryU(Y_cluster,k);
    Y_sample_temp = B3_old*U';
    [~,Y_sample] = max(Y_sample_temp,[],2);
end

function U = generateBinaryU(Y_cluster,c)
  [num,~]=size(Y_cluster);
  U = zeros(c,num);
  for i=1:c
      index = find(Y_cluster==i);
      U(i,index) = 1;
      clear index
  end
end

function BY = generateBinaryMatrix(y,c)
  [num,~]=size(y);
  BY = zeros(num,c);
  for i = 1:c
      index = find(y==i);
      BY(index,i)=1;
  end
  BY = sparse(BY);
end

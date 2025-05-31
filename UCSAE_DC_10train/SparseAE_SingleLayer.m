%% STEP 0: Initialization the first hidden layer
%% 该函数实现
tic
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
load gtpatch;%导入训练样本的patch
%% preprocessing the train data
fprintf('prepocessing the train data\n');
fprintf('STEP 0: Initialize the first hidden layer\n');
imageChannels = 191;     % number of channels (rgb, so 3)
patchDim =2;         % patch dimension
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
hiddenSizeL1 = 700;    % number of hidden units
sparsityParam = 0.6;   % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.1;	       % epsilon for ZCA whitening;
num_input=34243;
onepicnum=8;
numpatches=num_input*onepicnum;
patches1=zeros(patchDim*patchDim*imageChannels,numpatches);
for imageNum = 1:num_input
   [rowNum colNum] = size(gtpatch(:,:,1,imageNum));%这里是4*4
   for patchNum = 1:onepicnum%从每个patch中选取4个小块
       xPos = randi([1,rowNum-patchDim+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
       yPos = randi([1,colNum-patchDim+1]);
       temp=gtpatch(xPos:xPos+patchDim-1,yPos:yPos+patchDim-1,:,imageNum);
       reshapetemp=reshape(temp,patchDim*patchDim*imageChannels,1);
       patches1(:,(imageNum-1)*onepicnum+patchNum) =reshapetemp;
   end
end
clear gtpatch;
clear temp reshapetemp;
patches1=reshape(patches1,patchDim,patchDim,imageChannels,numpatches);
patches=zeros(patchDim,patchDim,imageChannels,numpatches*4);
patches(:,:,:,1:numpatches)=patches1;
patches(:,:,:,numpatches+1:numpatches*2)=imrotate(patches1,90);
patches(:,:,:,numpatches*2+1:numpatches*3)=imrotate(patches1,180);
patches(:,:,:,numpatches*3+1:numpatches*4)=imrotate(patches1,270);
clear patches1;
patches=reshape(patches,patchDim*patchDim*imageChannels,numpatches*4);
rand_patch=randperm(numpatches*4);
patches=patches(:,rand_patch);
patches=normalizeData(patches);
patches=ZCAwhitenData(patches);
save('patches','patches','-v7.3');
clear patches;
toc
%% train W
tic
fprintf('train W\n');
addpath minFunc/;
load patches;
thetaL1=linerdecodertrainW(visibleSizeL1,hiddenSizeL1,patches,2800,numpatches*4, sparsityParam);
WL1 = reshape(thetaL1(1:hiddenSizeL1*visibleSizeL1), hiddenSizeL1, visibleSizeL1);
bL1= thetaL1(2*hiddenSizeL1*visibleSizeL1+1:2*hiddenSizeL1*visibleSizeL1+hiddenSizeL1);
save WL1.mat WL1;
save bL1.mat bL1;
%displayColorNetwork(WL1(:,1:3*patchDim*patchDim)');
%print -djpeg weights.jpg
toc
%% convolve and pool data
fprintf('convolve and pool data\n');
imageChannels = 191;     % number of channels (rgb, so 3)
patchDim = 2;         % patch dimension
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
hiddenSizeL1 = 700;    % number of hidden units
sparsityParam = 0.6;   % 0.035 desired average activation of the hidden units.
imageDim=7;
poolDim=3;
bbb=floor((imageDim-patchDim+1)/poolDim);
PN=17150+17150;
load ZCAWhite;
load WL1;
load bL1;
% load inputpatch1;
load trainpatch_new;
load testpatch_new;
inputpatch1(:,:,:,1:3450)=trainpatch_new;
inputpatch1(:,:,:,3450+1:17150+17150)=testpatch_new;
trainImages=inputpatch1;%把输入图像归一到0.1-0.9，这是为了保证训练和编码的输入是一样的。

pooledFeaturesL1=zeros(hiddenSizeL1,PN,bbb,bbb);
[im1,in1,imagechanne1l]=size(trainImages(:,:,:,1));
tic;
for jj=1:PN
    image=trainImages(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite*testX;% 0.20s
    Z=sigmoid(WL1*testX+repmat(bL1,[1 size(testX,2)]));% 1.2s  
    Z=Z';
    Z=reshape(Z,im1-patchDim+1,in1-patchDim+1,size(WL1,1));% 0.23s
    Z1=permute(Z,[3 1 2]);%Z1的维数为hiddenSizeL1-convolvedDim-convolvedDim,Z1表示卷积后的影像，维数为500-610-340
    pooledFeaturesThis=cnnPool(poolDim,Z1);
    pooledFeaturesL1(:,jj,:,:)=pooledFeaturesThis;
    jj
end
toc;
numTrain=3450;
numTest=30850;
pooledFeaturesL1=permute(pooledFeaturesL1,[3 4 1 2]);
pooledFeatureTrain=pooledFeaturesL1(:,:,:,1:3450);
pooledFeatureTest=pooledFeaturesL1(:,:,:,3450+1:17150+17150);
pooledFeatureTrain=reshape(pooledFeatureTrain,numel(pooledFeatureTrain)/numTrain,numTrain);
pooledFeatureTest=reshape(pooledFeatureTest,numel(pooledFeatureTest)/numTest,numTest);

%% classification
tic
fprintf('classification\n');
softmaxLambda = 1e-4;
numClasses =6;
load train_test_label_DC1;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(pooledFeaturesL1)/PN, numClasses, softmaxLambda,pooledFeatureTrain, trainLabel_added_DC1, options);                       
[pred] = softmaxPredict(softmaxModel,pooledFeatureTest);
acc1 = (pred(:) == testLabel_added_DC1(:));
acc2 = sum(acc1) / size(acc1, 1);
fprintf('Accuracy: %2.3f%%\n', acc2 * 100);
toc

%% Whole image prediction
% load imagepatch;
% load ZCAWhite_s2;
% load WL1_s2;
% load bL1_s2;
% PN1=307*280;
% pooledFeatures_s2=zeros(hiddenSizeL1_s2,PN1,bbb_s2,bbb_s2);
% [im1,in1,imagechanne1l]=size(imagepatch(:,:,:,1));
% tic;
% for jj=1:PN1
%     image=imagepatch(:,:,:,jj);
%     convlocation=myconvlocation(image,patchDim_s2);
%     testX=(image(convlocation));% 0.14s
%     testX=ZCAWhite_s2*testX;% 0.20s
%     Z=sigmoid(WL1_s2*testX+repmat(bL1_s2,[1 size(testX,2)]));% 1.2s  
%     Z=Z';
%     Z=reshape(Z,im1-patchDim_s2+1,in1-patchDim_s2+1,size(WL1_s2,1));% 0.23s
%     Z1=permute(Z,[3 1 2]);%Z1的维数为hiddenSizeL1-convolvedDim-convolvedDim,Z1表示卷积后的影像，维数为500-610-340
%     pooledFeaturesThis=cnnPool(poolDim_s2,Z1);
%     pooledFeatures_s2(:,jj,:,:)=pooledFeaturesThis;
%     jj
% end
% toc;
% pooledFeatures_s2=permute(pooledFeatures_s2,[3 4 1 2]);
% pooledFeatures_s2=reshape(pooledFeatures_s2,numel(pooledFeatures_s2)/PN1,PN1);
% save pooledFeatures_s2 pooledFeatures_s2 '-v7.3';
% 
% [pred_allmap]=softmaxPredict(softmaxModel,pooledFeatures_s2);
% pred_allmap1=reshape(pred_allmap,280,307);
% figure;imagesc(pred_allmap1);
% enviwrite(pred_allmap1',307,280,1,'pred_allmap1');
% save pred_allmap1 pred_allmap1 '-v7.3';

%% STEP 0: Initialization the first hidden layer
%% 该函数实现
tic
fprintf('STEP 0: Initialize the first hidden layer\n');
imageChannels = 103;     % number of channels (rgb, so 3)
patchDim_s2 = 2;         % patch dimension
% numpatches =(25664+8557+8555)*8; 
visibleSizeL1_s2 = patchDim_s2 * patchDim_s2 * imageChannels;  % number of input units
hiddenSizeL1_s2 = 1100;    % number of hidden units
sparsityParam_s2 = 0.7;   % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.01;	       % epsilon for ZCA whitening
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
%load imagepatch;%导入训练样本的patch,25*25*6*450
%load gtpatch;
load trainpatch_new;
%% preprocessing the train data
fprintf('prepocessing the train data\n');
%num_input=340*610;
%num_input=42776;
num_input=4400;
onepicnum=40;
numpatches_s2=num_input*onepicnum;
patches1=zeros(patchDim_s2*patchDim_s2*imageChannels,numpatches_s2);
for imageNum = 1:num_input
   [rowNum colNum] = size(trainpatch_new(:,:,1,imageNum));%这里是4*4
   for patchNum = 1:onepicnum%从每个patch中选取4个小块
       xPos = randi([1,rowNum-patchDim_s2+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
       yPos = randi([1,colNum-patchDim_s2+1]);
       temp=trainpatch_new(xPos:xPos+patchDim_s2-1,yPos:yPos+patchDim_s2-1,:,imageNum);
       reshapetemp=reshape(temp,patchDim_s2*patchDim_s2*imageChannels,1);
       patches1(:,(imageNum-1)*onepicnum+patchNum) =reshapetemp;
   end
end
clear trainpatch_new;
clear temp reshapetemp;
patches1=reshape(patches1,patchDim_s2,patchDim_s2,imageChannels,numpatches_s2);
patches_s2=zeros(patchDim_s2,patchDim_s2,imageChannels,numpatches_s2*4);
patches_s2(:,:,:,1:numpatches_s2)=patches1;
patches_s2(:,:,:,numpatches_s2+1:numpatches_s2*2)=imrotate(patches1,90);
patches_s2(:,:,:,numpatches_s2*2+1:numpatches_s2*3)=imrotate(patches1,180);
patches_s2(:,:,:,numpatches_s2*3+1:numpatches_s2*4)=imrotate(patches1,270);
clear patches1;
patches_s2=reshape(patches_s2,patchDim_s2*patchDim_s2*imageChannels,numpatches_s2*4);
%patches_s2=patches1;
%patches_s2=reshape(patches_s2,patchDim_s2*patchDim_s2*imageChannels,numpatches_s2);
rand_patch_s2=randperm(numpatches_s2*4);
patches_s2=patches_s2(:,rand_patch_s2);
patches_s2=normalizeData_s2(patches_s2);
patches_s2=ZCAwhitenData_s2(patches_s2);
save('patches_s2','patches_s2','-v7.3');
toc
%% train W
tic
fprintf('train W\n');
addpath minFunc/;
thetaL1_s2=linerdecodertrainW_s2(visibleSizeL1_s2,hiddenSizeL1_s2,patches_s2,4000,numpatches_s2*4, sparsityParam_s2);
WL1_s2 = reshape(thetaL1_s2(1:hiddenSizeL1_s2*visibleSizeL1_s2), hiddenSizeL1_s2, visibleSizeL1_s2);
bL1_s2= thetaL1_s2(2*hiddenSizeL1_s2*visibleSizeL1_s2+1:2*hiddenSizeL1_s2*visibleSizeL1_s2+hiddenSizeL1_s2);
save WL1_s2.mat WL1_s2;
save bL1_s2.mat bL1_s2;
%displayColorNetwork(WL1(:,1:3*patchDim*patchDim)');
%print -djpeg weights.jpg
toc
%% convolve and pool data
fprintf('convolve and pool data\n');
imageDim=7;
poolDim_s2=3;
bbb_s2=floor((imageDim-patchDim_s2+1)/poolDim_s2);

PN=43000;
load ZCAWhite_s2;
load WL1_s2;
load bL1_s2;
load trainpatch_new;
load testpatch_new;
inputpatch1_s2(:,:,:,1:4400)=trainpatch_new;
inputpatch1_s2(:,:,:,4400+1:21500+21500)=testpatch_new;
% load inputpatch1;
trainImages_s2=inputpatch1_s2;%把输入图像归一到0.1-0.9，这是为了保证训练和编码的输入是一样的。
pooledFeaturesL1_s2=zeros(hiddenSizeL1_s2,PN,bbb_s2,bbb_s2);
[im1,in1,imagechanne1l]=size(trainImages_s2(:,:,:,1));
convolvedFeatures_s2=zeros(hiddenSizeL1_s2,imageDim-patchDim_s2+1,imageDim-patchDim_s2+1,PN);
tic;
for jj=1:PN
    image_s2=trainImages_s2(:,:,:,jj);
    convlocation_s2=myconvlocation(image_s2,patchDim_s2);
    testX_s2=(image_s2(convlocation_s2));% 0.14s
    testX_s2=ZCAWhite_s2*testX_s2;% 0.20s
    Z_s2=sigmoid(WL1_s2*testX_s2+repmat(bL1_s2,[1 size(testX_s2,2)]));% 1.2s  
    Z_s2=Z_s2';
    Z_s2=reshape(Z_s2,im1-patchDim_s2+1,in1-patchDim_s2+1,size(WL1_s2,1));% 0.23s
    Z1_s2=permute(Z_s2,[3 1 2]);%Z1?奈猦iddenSizeL1-convolvedDim-convolvedDim,Z1表示卷积后的影像，维数为500-610-340
    convolvedFeatures_s2(:,:,:,jj)=Z1_s2;
    pooledFeaturesThis_s2=cnnPool(poolDim_s2,Z1_s2);
    pooledFeaturesL1_s2(:,jj,:,:)=pooledFeaturesThis_s2;
%     pooledFeaturesL1(:,jj,:,:)=cnnPool(poolDim,Z1);
    jj
end
toc;
save convolvedFeatures_s2 convolvedFeatures_s2 '-v7.3';
numTrain_s2=4400;
numTest_s2=38600;
pooledFeaturesL1_s2=permute(pooledFeaturesL1_s2,[3 4 1 2]);
pooledFeatureTrain_s2=pooledFeaturesL1_s2(:,:,:,1:4400);
pooledFeatureTest_s2=pooledFeaturesL1_s2(:,:,:,4400+1:450+42550);
pooledFeatureTrain_s2=reshape(pooledFeatureTrain_s2,numel(pooledFeatureTrain_s2)/numTrain_s2,numTrain_s2);
pooledFeatureTest_s2=reshape(pooledFeatureTest_s2,numel(pooledFeatureTest_s2)/numTest_s2,numTest_s2);
save pooledFeatureTrain_s2 pooledFeatureTrain_s2 '-v7.3';
save pooledFeatureTest_s2 pooledFeatureTest_s2 '-v7.3';

%% Softmax classification
tic
fprintf('classification\n');
softmaxLambda = 1e-4;
numClasses =9;
load train_test_label_PU1;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(pooledFeaturesL1_s2)/PN, numClasses, softmaxLambda,pooledFeatureTrain_s2, trainLabel_added_PU1, options);                       
[pred_s2] = softmaxPredict(softmaxModel,pooledFeatureTest_s2);
acc1_s2 = (pred_s2(:) == testLabel_added_PU1(:));
acc2_s2 = sum(acc1_s2) / size(acc1_s2, 1);
fprintf('Accuracy: %2.3f%%\n', acc2_s2 * 100);
save acc2_s2 acc2_s2;

%% SVM Classification
%load train_test_label_PU1;
%model=svmtrain(trainLabel_added_PU1,pooledFeaturesTrain_s2','-t 2 -g 0.001 -c 10000');
%[predict_label_s2,accuracy]=svmpredict(testLabel_added_PU1,pooledFeaturesTest_s2',model);
%save predict_label_s2 predict_label_s2 '-v7.3';
%save accuracy_svm accuracy '-v7.3';

%% Whole Image Prediction
load imagepatch;
load ZCAWhite_s2;
trainImages1=imagepatch;
PN1=340*610;
pooledFeatures_all=zeros(hiddenSizeL1_s2,PN1,bbb_s2,bbb_s2);
[im1,in1,imagechanne1l]=size(trainImages1(:,:,:,1));
tic;
for jj=1:PN1
    image=trainImages1(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim_s2);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite_s2*testX;% 0.20s
    Z=sigmoid(WL1_s2*testX+repmat(bL1_s2,[1 size(testX,2)]));% 1.2s  
    Z=Z';
    Z=reshape(Z,im1-patchDim_s2+1,in1-patchDim_s2+1,size(WL1_s2,1));% 0.23s
    Z1=permute(Z,[3 1 2]);%Z1的维数为hiddenSizeL1-convolvedDim-convolvedDim,Z1表示卷积后的影像，维数为500-610-340
    pooledFeaturesThis=cnnPool(poolDim_s2,Z1);
    pooledFeatures_all(:,jj,:,:)=pooledFeaturesThis;
    jj
end
toc;
num_all=340*610;
pooledFeatures_all=permute(pooledFeatures_all,[3 4 1 2]);
pooledFeature_window=reshape(pooledFeatures_all,numel(pooledFeatures_all)/num_all,num_all);
save pooledFeature_window pooledFeature_window '-v7.3';
[pred_allmap]=softmaxPredict(softmaxModel,pooledFeature_window);
pred_allmap1=reshape(pred_allmap,610,340);
enviwrite(pred_allmap1',340,610,1,'pred_allmap1');
toc

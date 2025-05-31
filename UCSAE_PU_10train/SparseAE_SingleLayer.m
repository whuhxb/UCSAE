%% STEP 0: Initialization the first hidden layer
%% �ú���ʵ��
tic
fprintf('STEP 0: Initialize the first hidden layer\n');
imageChannels = 103;     % number of channels (rgb, so 3)
patchDim = 2;         % patch dimension
% numpatches =(25664+8557+8555)*8; 
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
hiddenSizeL1 = 1100;    % number of hidden units
sparsityParam = 0.7;   % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.01;	       % epsilon for ZCA whitening
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
load gtpatch;%����ѵ��������patch,25*25*6*450
%% preprocessing the train data
fprintf('prepocessing the train data\n');
num_input=42776;
onepicnum=8;
numpatches=num_input*onepicnum;
patches1=zeros(patchDim*patchDim*imageChannels,numpatches);
for imageNum = 1:num_input
   [rowNum colNum] = size(gtpatch(:,:,1,imageNum));%������4*4
   for patchNum = 1:onepicnum%��ÿ��patch��ѡȡ4��С��
       xPos = randi([1,rowNum-patchDim+1]);%�����ȡһ�����꣬����randi([a,b])��ʾ���ȡһ����ΧΪ[a,b]������
       yPos = randi([1,colNum-patchDim+1]);
       temp=gtpatch(xPos:xPos+patchDim-1,yPos:yPos+patchDim-1,:,imageNum);
       reshapetemp=reshape(temp,patchDim*patchDim*imageChannels,1);
       patches1(:,(imageNum-1)*onepicnum+patchNum) =reshapetemp;
   end
end
clear gtpatch;
clear temp reshapetemp;
patches1=normalizeData(patches1);
patches1=ZCAwhitenData(patches1);
save('patches1','patches1','-v7.3');
toc
%% train W
tic
fprintf('train W\n');
addpath minFunc/;
thetaL1=linerdecodertrainW(visibleSizeL1,hiddenSizeL1,patches1,4000,numpatches, sparsityParam);
WL1 = reshape(thetaL1(1:hiddenSizeL1*visibleSizeL1), hiddenSizeL1, visibleSizeL1);
bL1= thetaL1(2*hiddenSizeL1*visibleSizeL1+1:2*hiddenSizeL1*visibleSizeL1+hiddenSizeL1);
save WL1.mat WL1;
save bL1.mat bL1;
displayColorNetwork(WL1(:,1:3*patchDim*patchDim)');
print -djpeg weights.jpg
toc
%% convolve and pool data
fprintf('convolve and pool data\n');
imageDim=7;
poolDim=3;
bbb=floor((imageDim-patchDim+1)/poolDim);

PN=43000;
load ZCAWhite;
load WL1;
load bL1;
meanPatch=zeros(visibleSizeL1,1);
load trainpatch_new;
load testpatch_new;
inputpatch1(:,:,:,1:4400)=trainpatch_new;
inputpatch1(:,:,:,4400+1:21500+21500)=testpatch_new;
% load inputpatch1;
trainImages=inputpatch1;%������ͼ���һ��0.1-0.9������Ϊ�˱�֤ѵ���ͱ����������һ���ġ�
pooledFeaturesL1=zeros(hiddenSizeL1,PN,bbb,bbb);
[im1,in1,imagechanne1l]=size(trainImages(:,:,:,1));
convolvedFeatures=zeros(hiddenSizeL1,imageDim-patchDim+1,imageDim-patchDim+1,PN);
tic;
for jj=1:PN
    image=trainImages(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite*testX;% 0.20s
    Z=sigmoid(WL1*testX+repmat(bL1,[1 size(testX,2)]));% 1.2s  
    Z=Z';
    Z=reshape(Z,im1-patchDim+1,in1-patchDim+1,size(WL1,1));% 0.23s
    Z1=permute(Z,[3 1 2]);%Z1��ά��ΪhiddenSizeL1-convolvedDim-convolvedDim,Z1��ʾ������Ӱ��ά��Ϊ500-610-340
    convolvedFeatures(:,:,:,jj)=Z1;
    pooledFeaturesThis=cnnPool(poolDim,Z1);
    pooledFeaturesL1(:,jj,:,:)=pooledFeaturesThis;
%     pooledFeaturesL1(:,jj,:,:)=cnnPool(poolDim,Z1);
    jj
end
toc;
save convolvedFeatures convolvedFeatures '-v7.3';
numTrain=4400;
numTest=38600;
pooledFeaturesL1=permute(pooledFeaturesL1,[3 4 1 2]);
pooledFeatureTrain=pooledFeaturesL1(:,:,:,1:4400);
pooledFeatureTest=pooledFeaturesL1(:,:,:,4400+1:450+42550);
pooledFeatureTrain=reshape(pooledFeatureTrain,numel(pooledFeatureTrain)/numTrain,numTrain);
pooledFeatureTest=reshape(pooledFeatureTest,numel(pooledFeatureTest)/numTest,numTest);
%% classification
tic
fprintf('classification\n');
softmaxLambda = 1e-4;
numClasses =9;

load train_test_label_PU1;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(pooledFeaturesL1)/PN, numClasses, softmaxLambda,pooledFeatureTrain, trainLabel_added_PU1, options);                       
[pred] = softmaxPredict(softmaxModel,pooledFeatureTest);
acc1 = (pred(:) == testLabel_added_PU1(:));
acc2 = sum(acc1) / size(acc1, 1);
fprintf('Accuracy: %2.3f%%\n', acc2 * 100);
toc

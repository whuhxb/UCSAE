load train_test_label_PU1.mat;
load pooledFeatureTrain_s2.mat;
load pooledFeatureTest_s2.mat;
model=svmtrain(trainLabel_added_PU1,pooledFeatureTrain_s2','-t 2 -g 0.001 -c 10000');
[predict_label_s2,accuracy]=svmpredict(testLabel_added_PU1,pooledFeatureTest_s2',model);
save predict_label_s2 predict_label_s2 '-v7.3';
save acc_svm accuracy '-v7.3';


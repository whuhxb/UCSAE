load train_test_label_DC1;
load pooledFeatureTrain;
load pooledFeatureTest;
model=svmtrain(trainLabel_added_DC1,pooledFeatureTrain','-t 2 -g 0.052 -c 16');
[predict_label_s2,accuracy]=svmpredict(testLabel_added_DC1,pooledFeatureTest',model);
save predict_label_s2 predict_label_s2 '-v7.3';

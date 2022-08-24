clear all
close all
clc
%% Build the datasets
load('dp_PSDfeature_2.mat')

%Class 1: Begin MI
%Class 2: END MI

%BEGIN MI
class1 = PSD_bm_fs(:,:);

%END MI
class2 = PSD_em_fs(:,:);

% Begin MI REST
rest_bm_class = PSD_bm_rst_fs(:,:);

% End MI REST
rest_em_class = PSD_em_rst_fs(1:size(class2,1),:);

%class 1 for em
class1_em = PSD_bm_em(:,:); 

%class 2 for bm
class2_bm = PSD_em_bm(:,:); 


%% Seperate features

% Class1 vs Rest dataset
Data_Class1vsRest= [class1; rest_bm_class];
Labels_Class1vsRest = [ones(1,length(class1)) 2*ones(1,length(rest_bm_class))];

% Class2 vs Rest dataset
Data_Class2vsRest = [class2; rest_em_class];
Labels_Class2vsRest = [ones(1,length(class2)) 2*ones(1,length(rest_em_class))];

%Class1 vs Class2
Data_Class1vsClass2 = [class1_em(1:600,:); class2_bm];
Labels_Class1vsClass2 = [ones(1,length(class1_em(1:600,:))) 2*ones(1,length(class2_bm))];


%% Saving data/labels in CSV
%Class 1 vs Rest
feat_label_C1Rest(:,:) = Data_Class1vsRest; 
feat_label_C1Rest(:,6) = Labels_Class1vsRest'; 

%write features and labels to csv file 
%writematrix(feat_label_C1Rest,'Sub4_wH_C1R_data.csv') 

%Class 2 vs Rest
feat_label_C2Rest(:,:) = Data_Class2vsRest; 
feat_label_C2Rest(:,6) = Labels_Class2vsRest; 

%Class 1 vs Class 2
feat_label_C1C2(:,:) = Data_Class1vsClass2; 
feat_label_C1C2(:,6) = Labels_Class1vsClass2'; 

%write features and labels to csv file 
%writematrix(feat_label_C2Rest,'Sub5_wH_C2Rdata.csv') 


%% dLDA - Classification 

%[trainedClassifier, validationAccuracy,trueLabels, predLabels] = AdaBoostTree(feat_label_C2Rest); 

[trainedClassifier_dLDA_1, validationAccuracy1,trueLabels1, predLabels1] = dLDA(feat_label_C1Rest); 
[trainedClassifier_dLDA_2, validationAccuracy2,trueLabels2, predLabels2] = dLDA(feat_label_C2Rest); 
[trainedClassifier_dLDA_3, validationAccuracy3,trueLabels3, predLabels3] = dLDA(feat_label_C1C2); 


%% dLDA - Results 

figure('units','normalized','Position',[0.1,0.1,0.5,0.5])
tiledlayout(3,1)

nexttile
cm = confusionmat(predLabels1,trueLabels1); 
confusionchart(cm, ["Begin MI", "Rest"], 'FontSize', 18);
title('Onset: Begin MI vs. REST');


nexttile
cm2 = confusionmat(predLabels3,trueLabels3); 
confusionchart(cm2, ["Begin MI", "END MI"], 'FontSize', 18);
title('Termination: Begin MI vs. End MI');

nexttile
cm1 = confusionmat(predLabels2,trueLabels2); 
confusionchart(cm1, ["END MI", "Rest"], 'FontSize', 18);
title('End MI vs. REST');


sgtitle('Subject 4 with Harmony: Confusion Matrix(dLDA) of PSD features', 'FontSize', 18)


%Accuracies 
disp('dLDA Accuracy')
acc = ['Accuracy: Class 1 vs. Rest: ', num2str(validationAccuracy1*100)];
disp(acc)

acc2 = ['Accuracy: Class 1 vs. Class 2: ', num2str(validationAccuracy3*100)];
disp(acc2)

acc1 = ['Accuracy: Class 2 vs. Rest: ', num2str(validationAccuracy2*100)];
disp(acc1)
disp(' ')

%% AdaBoostTree - Classification 

%[trainedClassifier, validationAccuracy,trueLabels, predLabels] = AdaBoostTree(feat_label_C2Rest); 

[trainedClassifier_ADA_1, validationAccuracy1,trueLabels1, predLabels1] = AdaBoostTree(feat_label_C1Rest); 
[trainedClassifier_ADA_2, validationAccuracy2,trueLabels2, predLabels2] = AdaBoostTree(feat_label_C2Rest); 
[trainedClassifier_ADA_3, validationAccuracy3,trueLabels3, predLabels3] = AdaBoostTree(feat_label_C1C2); 


%% AdaBoostTree - Results 

figure('units','normalized','Position',[0.1,0.1,0.5,0.5])
tiledlayout(3,1)

nexttile
cm = confusionmat(predLabels1, trueLabels1); 
confusionchart(cm, ["Begin MI", "Rest"], 'FontSize', 18);
title('Onset: Begin MI vs. REST');

nexttile
cm2 = confusionmat(predLabels3, trueLabels3); 
confusionchart(cm2, ["Begin MI", "END MI"], 'FontSize', 18);
title('Termination: Begin MI vs. End MI');

nexttile
cm1 = confusionmat(predLabels2, trueLabels2); 
confusionchart(cm1, ["END MI", "Rest"], 'FontSize', 18);
title('End MI vs. REST');


sgtitle('Subject 4 with Harmony: Confusion Matrix(AdaBoost) of PSD features', 'FontSize', 18)


%Accuracies 
disp('ADABoost Accuracy')
acc = ['Accuracy: Class 1 vs. Rest: ', num2str(validationAccuracy1*100)];
disp(acc)

acc2 = ['Accuracy: Class 1 vs. Class 2: ', num2str(validationAccuracy3*100)];
disp(acc2)

acc1 = ['Accuracy: Class 2 vs. Rest: ', num2str(validationAccuracy2*100)];
disp(acc1)
disp(' ')


%% KNN - Classification 

%[trainedClassifier, validationAccuracy,trueLabels, predLabels] = AdaBoostTree(feat_label_C2Rest); 

[trainedClassifier_KNN_1, validationAccuracy1,trueLabels1, predLabels1] = KNN(feat_label_C1Rest); 
[trainedClassifier_KNN_2, validationAccuracy2,trueLabels2, predLabels2] = KNN(feat_label_C2Rest); 
[trainedClassifier_KNN_3, validationAccuracy3,trueLabels3, predLabels3] = KNN(feat_label_C1C2); 


%% KNN - Results 

figure('units','normalized','Position',[0.1,0.1,0.5,0.5])
tiledlayout(3,1)

nexttile
cm = confusionmat(predLabels1,trueLabels1); 
confusionchart(cm, ["Begin MI", "Rest"], 'FontSize', 18);
title('Onset: Begin MI vs. REST');

nexttile
cm2 = confusionmat(predLabels3,trueLabels3); 
confusionchart(cm2, ["Begin MI", "END MI"], 'FontSize', 18);
title('Termination: Begin MI vs. End MI');

nexttile
cm1 = confusionmat(predLabels2,trueLabels2); 
confusionchart(cm1, ["END MI", "Rest"], 'FontSize', 18);
title('End MI vs. REST');


sgtitle('Subject 4 without Harmony: Confusion Matrix(KNN) of PSD features', 'FontSize', 18)


%Accuracies 
disp('KNN Accuracy')
acc = ['Accuracy: Class 1 vs. Rest: ', num2str(validationAccuracy1*100)];
disp(acc)

acc2 = ['Accuracy: Class 1 vs. Class 2: ', num2str(validationAccuracy3*100)];
disp(acc2)

acc1 = ['Accuracy: Class 2 vs. Rest: ', num2str(validationAccuracy2*100)];
disp(acc1)
disp(' ')

%% SVM - Classification 

%[trainedClassifier, validationAccuracy,trueLabels, predLabels] = AdaBoostTree(feat_label_C2Rest); 

[trainedClassifier_SVM_1, validationAccuracy1,trueLabels1, predLabels1] = SVM(feat_label_C1Rest); 
[trainedClassifier_SVM_2, validationAccuracy2,trueLabels2, predLabels2] = SVM(feat_label_C2Rest); 
[trainedClassifier_SVM_3, validationAccuracy3,trueLabels3, predLabels3] = SVM(feat_label_C1C2); 


%% SVM - Results 

figure('units','normalized','Position',[0.1,0.1,0.5,0.5])
tiledlayout(3,1)

nexttile
cm = confusionmat(predLabels1,trueLabels1); 
confusionchart(cm, ["Begin MI", "Rest"], 'FontSize', 18);
title('Onset: Begin MI vs. REST');

nexttile
cm2 = confusionmat(predLabels3,trueLabels3); 
confusionchart(cm2, ["Begin MI", "END MI"], 'FontSize', 18);
title('Termination: Begin MI vs. End MI');

nexttile
cm1 = confusionmat(predLabels2,trueLabels2); 
confusionchart(cm1, ["END MI", "Rest"], 'FontSize', 18);
title('End MI vs. REST');


sgtitle('Subject 3 with Harmony: Confusion Matrix(SVM) of PSD features', 'FontSize', 18)


%Accuracies 
disp('SVM Accuracy')
acc = ['Accuracy: Class 1 vs. Rest: ', num2str(validationAccuracy1*100)];
disp(acc)

acc2 = ['Accuracy: Class 1 vs. Class 2: ', num2str(validationAccuracy3*100)];
disp(acc2)

acc1 = ['Accuracy: Class 2 vs. Rest: ', num2str(validationAccuracy2*100)];
disp(acc1)
disp(' ')



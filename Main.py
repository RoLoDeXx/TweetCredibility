
import numpy as np
import PreProcessing as prp
import NB_Classfier as nb
import SVM_Classifier as svm
import Bar_graph_plot as br


if __name__ == "__main__":
    training_data = prp.Data_preprocessing('RawTrainingDataSet.csv', 'Training')
    test_data = prp.Data_preprocessing('RawTestDataSet.csv', 'Test')
    SvmClassifier = svm.SVMClassifier('Training_feature_extracted.csv')
    nbclassifier = nb.NbClassifier('Training_feature_extracted.csv')

    test_point = np.array([5, 8, 3])
    test_point = test_point.reshape(1, 3)
    predicted_class = SvmClassifier.classify(test_point)
    Pred_test_data_class = SvmClassifier.classify_testdata('Test_feature_extracted.csv')
    accuracy_svm=SvmClassifier.confusionMatrix(Pred_test_data_class)

    a, accuracy_nb = nbclassifier.classify_all('Test_feature_extracted.csv')

    br.bar_plot(accuracy_svm,accuracy_nb)


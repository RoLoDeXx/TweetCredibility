import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import *
from sklearn import svm
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, confusion_matrix
import scikitplot.plotters as skplt



warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', 30000)
My_col=['Retweets','Favorites','New_Feature','Class']


class SVMClassifier(object):

    def __init__(self, file_name):
        #Load traingin csv
        self.Processed_file = pd.read_csv(file_name, sep=',', usecols=My_col, index_col=None)

        # Create arbitrary dataset for example
        df = pd.DataFrame({'x': self.Processed_file['Retweets'],
                           'y': self.Processed_file['Favorites'],
                           'Class': self.Processed_file['Class']}
                          )
        X = np.array(self.Processed_file.values[:, :3])
        Y = self.Processed_file['Class']
        #init the model
        self.SVM = svm.SVC(kernel='linear', C=1.0, gamma=2)
        self.SVM.fit(X, Y.values)

    def classify_testdata(self, filename):
        self.test_file = pd.read_csv(filename, sep=',', usecols=My_col, index_col=None)
        self.test = np.array([self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature']])
        self.test = np.array(self.test_file.values[:, :3])
        self.predicted_label = self.SVM.predict(self.test)
        return self.predicted_label

    def classify(self, x):
        output = self.SVM.predict(x)
        return output


    def confusionMatrix(self, predict):
        Accuracy_Score = accuracy_score(self.test_file['Class'], predict)
        print("Accuracy for SVM")
        accuracy=Accuracy_Score*100
        print(accuracy)
        print("Confusion Matrix for SVM")
        #print(confusion_matrix(self.test_file['Class'], predict))
        skplt.plot_confusion_matrix(self.test_file['Class'], predict)
        plt.show()
        return accuracy


if __name__ == "__main__":
    print("You are in main")

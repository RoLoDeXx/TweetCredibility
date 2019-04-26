import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import scipy.stats as stats
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
import scikitplot.plotters as skplt

pd.set_option('display.max_colwidth', 30000)
cols=['Class']
mycol=['Retweets','Favorites','New_Feature']
class NbClassifier:
   def __init__(self,file_name):
       self.file = pd.read_csv(file_name, sep=',')
       X = np.array(self.file.values[:, :3])
       x = self.file.Class
       self.NB = GaussianNB()
       self.NB.fit(X, x)


   def classify_all(self,filename):
        self.test_file = pd.read_csv(filename, sep=',', index_col=None)
        test = np.array(self.test_file.values[:, :3])
        test_data_class = self.test_file.Class
        self.output = self.NB.predict(test)
        probability = self.NB.predict_proba(test)
        cm = metrics.confusion_matrix(test_data_class, self.output)
        accuracy = accuracy_score(test_data_class, self.output)
        print("Accuracy for Naive Bayes")
        print(accuracy*100)
        print("Confusion Matrix for Naive Bayes")
        #print(cm)
        skplt.plot_confusion_matrix(test_data_class, self.output)
        plt.show()
        return self.output, accuracy * 100

   def classify(self, x):
       output = self.NB.predict(x)
       probability = self.NB.predict_proba(x)
       return output, probability

   def plot_a(self):
       color = ['red' if l == 1 else 'green' for l in self.file['Class']]
       color_test = ['black' if l == 1 else 'blue' for l in self.output]
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(self.file['Retweets'], self.file['Favorites'], self.file['New_Feature'],
                  zdir='z', s=20, depthshade=True, color=color, marker='^')
       ax.scatter(self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature'], zdir='z',
                  s=20, depthshade=True, color=color_test, marker='^')
       plt.title("NB Classifier")
       ax.set_xlabel('X axis')
       ax.set_ylabel('Y axis')
       ax.set_zlabel('Z axis')
       ax.legend(loc=2)
       plt.show()

if __name__ == "__main__":
     print("You are in main")



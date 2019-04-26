import matplotlib.pyplot as plt;
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


class bar_plot(object):

    def __init__(self,svm,nb):
        objects = ('SVM','NB')
        color=('red','blue')
        y_pos = np.arange(len(objects))
        performance = [svm,nb] ## Accuracy scores for both classifiers
        plt.bar(y_pos, performance, align='center', alpha=0.5,color=color)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.title('Comparison of Accuracies')
        plt.show()

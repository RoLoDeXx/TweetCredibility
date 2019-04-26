import pandas as pd
from string import whitespace
import os
import warnings

warnings.filterwarnings("ignore")


pd.set_option('display.max_colwidth', 30000)
My_col = ['Date', 'Tweet_Text', 'Tweet_Id', 'User_Id', 'User_Name', 'User_Screen_Name', 'Retweets', 'Favorites',
          'Class']

class Data_preprocessing(object):


    def __init__(self, file_name, type):
        try:
            Raw_file = pd.read_csv(file_name, sep=',', usecols=My_col, index_col=None, quoting=3, encoding='utf-8')
            self.Data_preprocessed_file = Raw_file.dropna(subset=My_col)
        except (FileNotFoundError, FileExistsError, MemoryError) as e:
            print("file is not in correct format")

        try:
            prefix = '_Cleaned.csv'
            file = type + prefix
            self.Data_preprocessed_file.to_csv(file, header=None, sep=',', index=False)
        except PermissionError:
            print("file is opened by someone, please rerun after closing the file")
        self.Data_preprocessed_file['Date'] = self.Data_preprocessed_file['Date'].astype(str)

        def process_date(x):
            try:
                return x[x.find("[") + 1: x.find("]") - 1]
            except (ValueError, SyntaxError) as e:
                print("Check the data in a file")

        try:
            self.Data_preprocessed_file['Date'] = self.Data_preprocessed_file['Date'].map(lambda x: process_date(x))
        except(TypeError, SyntaxError, SystemExit, SyntaxWarning) as e:
            print("Check the data in a file")

        try:
            self.Data_preprocessed_file = self.Data_preprocessed_file.sort_values(by='Date')
        except ValueError:
            print("Values are not in date format")

        self.Data_preprocessed_file = self.Data_preprocessed_file[(self.Data_preprocessed_file['Date'] > '2011-07-31 23:59:59') & (self.Data_preprocessed_file['Date'] < '2011-08-30 00:00:00')]
        self.Data_preprocessed_file['Tweet_Text'] = self.Data_preprocessed_file['Tweet_Text'].astype(str)
        self.Data_preprocessed_file['Tweet_length'] = self.Data_preprocessed_file['Tweet_Text'].map(
            lambda x: len(x.translate(str.maketrans('', '', whitespace))))
        self.Data_preprocessed_file['Number_of_URL'] = self.Data_preprocessed_file['Tweet_Text'].map(
            lambda x: x.count('http*'))
        self.Data_preprocessed_file['No_of_@_word'] = self.Data_preprocessed_file['Tweet_Text'].map(lambda x: x.count('@'))
        self.Data_preprocessed_file['No_of_hash_word'] = self.Data_preprocessed_file['Tweet_Text'].map(
            lambda x: x.count('#'))
        self.Data_preprocessed_file['Length_of_User_Name'] = self.Data_preprocessed_file['User_Screen_Name'].map(
            lambda x: len(str(x)))

        Spam_count = {}

        def Spam_word_count(Word):
            Spam_list = ['Zoo', 'Tiger', 'Little girl','penguin']
            for i in Spam_list:
                try:
                    Spam_count[i] = Word.count(i)
                except ValueError:
                    print("Cant find the word list as a parameter")
            return sum(Spam_count.values())

        self.Data_preprocessed_file['Number_of_Spam_Word'] = self.Data_preprocessed_file['Tweet_Text'].map(
            lambda x: Spam_word_count(x.split(' ')))

        Swear_count = {}

        def Swear_word_count(Word):
            Swear_list = ['Burnt', 'Racism','Gun','Missiles','Suck','Fuck','Fucked','Rape','Racist','Firework']
            for i in Swear_list:
                try:
                    Swear_count[i] = Word.count(i)
                except ValueError:
                    print("Cant find the word list as a parameter")
            return sum(Swear_count.values())

        self.Data_preprocessed_file['Number_of_Swear_Word'] = self.Data_preprocessed_file['Tweet_Text'].map(
            lambda x: Swear_word_count(x.split(' ')))
        self.Data_preprocessed_file['New_Feature'] = self.Data_preprocessed_file['No_of_hash_word'] + \
                                                     self.Data_preprocessed_file['No_of_@_word'] + \
                                                     self.Data_preprocessed_file['Number_of_URL'] + \
                                                     self.Data_preprocessed_file['Number_of_Swear_Word'] + \
                                                     self.Data_preprocessed_file['Number_of_Spam_Word']

        try:
            prefix_pre = '_feature_selected.csv'
            file_name = type + prefix_pre
            self.Data_preprocessed_file.to_csv(file_name, sep=',', index=False)
        except PermissionError:
            print("file is opened by someone, please rerun after closing the file")


    #def feature_extraction(self, type):

        header = ["Retweets", "Favorites", "New_Feature", "Class"]
        try:
            prefix = '_feature_extracted.csv'
            file_name = type + prefix
            self.Data_preprocessed_file.to_csv(file_name, columns=header, sep=',', index=False)
        except PermissionError:
            print("file is opened by someone, please rerun after closing the file")


if __name__ == "__main__":
    training=Data_preprocessing('RawTrainingDataSet.csv','Training')
    test=Data_preprocessing('RawTestDataSet.csv','Test')


import pandas as pd
import numpy as np
import sklearn.model_selection 
import week4.class_wrappers


if __name__ == '__main__':
    path = 'data\\winequality-white.csv'
    df = pd.read_csv(path, delimiter = ';')
    df['good'] = df['quality'] > 5
    

features = df.drop(columns = ['good', 'quality']).values
labels = df['good'].values

train_features, test_features, train_labels, test_labels =\
    sklearn.model_selection.train_test_split(features, labels, test_size=0.3)

cw = week4.class_wrappers.ClassifierWrapper()
cw.train(train_features, train_labels)
print(cw.assess(test_features, test_labels, 'percent_correct'))
cw.save('data\\wine_model.xgb')
#test_preds = test_preds[:,1]>=0.5
#test_labels = test['good'].values

#correct =  1 - np.abs(test_preds.astype(int) - test_labels.astype(int))
# print (correct.sum()/len(correct))
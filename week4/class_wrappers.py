#We'll use classes to abstract away loading in data and saving it
#Classes group together variable and operations on those variables
#Keep classes relatively small
#Keep them operating on linked ideas
import os.path
import numpy as np
import sklearn.metrics
import sklearn.tree
import xgboost as xgb
import yaml



class ClassifierWrapper:
    def __init__(self):
        #Define EVERY variable you will use between your methods
        self.model = None
        self.mins_maxes = None
        self.pars = {
            'n_training_points': 0,
            'model_type': 'xgboost'
        }

    def train(self, x_data: np.ndarray, labels: np.ndarray, decision_tree: bool = False):
        self.pars['n_training_points'] = len(labels)
        if decision_tree:
            self.pars['model_type'] = 'decision tree'
            self.model = sklearn.DecisionTreeClassifier()
        else:
            self.model = xgb.XGBClassifier()
        self.model.fit(x_data, labels)

    def apply(self, x_data: np.ndarray):
       return self.model.predict_proba(x_data)

    def assess(self, x_data: np.ndarray, labels: np.ndarray, assesment: str):
        binary_pred = self.model.predict(x_data)
        if assesment == 'percent_correct':
            correct_points = 1 - np.abs(binary_pred.astype(int) - labels.astype(int))
            percent_correct = correct_points.sum()/len(correct_points)
            return percent_correct
        else:
            return sklearn.metrics.confusion_matrix(labels, binary_pred)
            # TP = cm[0][0]
            # FP = cm[0][1]
            # FN = cm[1][0]
            # TN = cm[1][1]       
        

    def save(self, path:str):
        if self.pars['model_type'] != 'xgboost':
            raise NotImplementedError('We can only save xgboost classifiers')

        base, ext = os.path.splitext(path)
        self.model.save_model(base + '.xgb')
        with open(base + '.yaml', 'w') as fo:
            yaml.dump(self.pars, fo)       

    def load(self, path: str):
        base, ext = os.path.splitext(path)
        self.model.load_model(base + '.xgb')
        with open(base + '.yaml', 'r') as fo:
            self.pars = yaml.load(fo.read())

    if __name__ == '__main__':
        cw = ClassifierWrapper()
        cw.train()
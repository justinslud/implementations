import numpy as np
from binarytree import my_BinaryTree

# require that all inputs be numeric and outputs be text or numeric
# assume binary classifications
        
class my_DecisionTree:
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def information_entropy(self, ps):
        if 0 in ps:
            return 0
        
        return sum(-p*np.log2(p) for p in ps)

    def information_gain(self, col_entropy, ps, es):
        return col_entropy - np.dot(ps, es)

    def build_tree(self, X_train, y_train):
        
        
    def fit(self, X_train, y_train):

        self.rows, self.cols = X_train.shape

        self.tree = BinaryTree()

        # X_train = np.delete(X_train, best_column, axis=1)
        # recursively complete tree with
        # [1, [2, [4, 3],

        # while X_train.shape[0] != 0:
        #   self.tree.insert_left((best_column))
        #   X_train = np.delete(X_train, best_column, axis=1)
        
        igs = []

        categories, counts = np.unique(y_train, return_counts=True)
    
##        print(categories, col_entropy)
##        print(type(col_entropy))
        yps = counts / rows

        entropy = self.information_entropy(yps)

        if self.verbose: print('{:20s}'.format('y probabilities'), yps, '\nentropy:', entropy)
        
        for num_col in range(cols):

            if self.verbose: print('\ncolumn number:', num_col)

            es = []
            
            attributes, attribute_counts = np.unique(X_train[:, num_col], return_counts=True)

            attribute_ps = attribute_counts / rows

            if self.verbose: print('attributes:', attributes, '\nattribute probabilities:', attribute_ps)
            
            for attribute in attributes:
                
                if self.verbose: print('attribute value:', attribute)
                
                ps = []
                for category in categories:
                    
                    matching_rows = len(np.where((X_train[:, num_col] == attribute) & (y_train == category))[0])
                    #gains = information_gain(X_train[np.where(y_train == val and X_train[num_col] == val)])

                    if self.verbose: print('y value:', category, 'attribute/y rows:', matching_rows)
                    ps.append(matching_rows)

                ps = np.array(ps) / len(np.where(X_train[:, num_col] == attribute)[0])

                if self.verbose: print('match probabilities:', ps)

                es.append(self.information_entropy(ps))
                if self.verbose: print('attribute entropy:', es[-1])

            igs.append(self.information_gain(entropy, attribute_ps, es))
            if self.verbose: print('column information gain:', igs[-1])

        best_column = np.array(igs).argmin()
        # X_train = np.delete(X_train, best_column, axis=1)
        # recursively complete tree with
        # [1, [2, [4, 3],

        # while X_train.shape[0] != 0:
        #   tree.append(build_tree(X_train, y_train))
            
    def predict(self, X_test):
        pass

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
y_train = np.array([0, 0, 0, 0, 0, 1])

dt = my_DecisionTree(verbose=True)
dt.fit(X_train, y_train)

class my_RandomForestClassifier:

    def __init__(self, num_trees: int, num_features: int, num_samples: int):
        self.trees = []


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        for i in range(M):
            X_sample, y_sample = X_train.sample
            tree = my_DecisionTree().fit()


    def predict(self, X_test):
        predictions = []
        
        for tree in self.trees:
            predictions.append(tree.predict(X_test))
        

    def score(self, X_test, y_test, metric='accuracy'):
        y_pred = self.predict(X_test)
        return sum(y_test[i] == y_pred[i] for i in range(len(X_test)))

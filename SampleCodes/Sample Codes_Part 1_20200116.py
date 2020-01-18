#########################################################################################
#############          Sample Code in the Book: Introduction to ML          #############
#########################################################################################

# Part 1: Introduction #

# 1. numpy, scipy, array, sparse array, eye, coordinate matrix format
import numpy as np
x = np.array([[1,2,3],[4,5,6]])   # shape is (2,3)
X_new = np.array([[5, 2.9, 1, 0.2]])  # create a (1,4) array
X_new[0,1]

from scipy import sparse
eye = np.eye(4)

# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)

# 2. matplotlib, pandas, 
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = 'x')

data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
		}
df = pd.DataFrame(data)
display(data)   # better format than print()

# 3. check version of python and library
import sys
sys.version  # python version

import pandas as pd
pd.__version__

import matplotlib as mpl
mpl.__version__


# 4. Iris data
from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_dataset.keys()  
# rst: dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
iris_dataset['target_names']
iris_dataset['feature_names']

# 5. Training and testing split
from sklearn.model_selection import train_test_split
x_train, x_text, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# 6. viz iris_dataset
iris_dataframe = pd.DataFrame(x_train['data'], columns = iris_dataset['feature_names'])

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize = (15,15),
                           marker = 'o', hist_kwds={'bins':20}, s=60,
						   alpha = 0.8, cmap=mglearn.cm3) # color map

# 7. knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])   # one row must be a [] inside a the overall list []
prediction = knn.predict(X_new)
pretictedTargetName = iris_dataset['target_names'][prediction]

y_pred = knn.predict(X_test)
print('Test Score is {:.2f}'.format(np.mean(y_pred==y_test)))
print('Test Score is {:.2f}'.format(knn.score(X_test, y_test)))

















#########################################################################################
#############          Sample Code in the Book: Introduction to ML          #############
#########################################################################################

# Part 2: Supervised learning #
np.bincount()  # count the non-negative int in an array

# 1. knn has both KNeighborsClassifier and KNeighborsregressor
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.model_selection import train_test_split
reg = knr(n_neighbors = 3)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Test set R^2 is {:.2f}".format(reg.score(X_test, y_test)))

fig, axes = plt.subplot(1, 3, figzie=(15,4))
for n_neighbors, ax in zip([1,3,9], axes):
	reg = knr(n_neighbors = n_neighbors)
	reg.fit(X_train, y_train)
	ax.plot(X_test, reg.predict(X_test))
	ax.plot(X_train, reg.predict(X_train))

	ax.set_title()
axes[0].legend(["test","train"], loc="best")

# 2. linear regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr = LinearRegression().fit(X_train, y_train)
lr.coef_, lr.intercept_

rg = Ridge().fit(alpha=0.1)
ls = Lasso().fit(alpha=0.01, max_iter=100000)

# 3. Linear model for classification: svc, LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

model.__class__.__name__  # get the model name
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) #stratify 根据y的原始比例，分配给train and test
logreg = LogisticRegression(max_iter=2000, C = 100).fit(X_train, y_train)  # C: regularization factor
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
			
# 4.1 Decision tree classifier
from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

  # see the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = "tree.dot", class_names = ["class 1","class 2"],
		feature_names = ...., impurity=False, filled=True)
import graphviz
with open("tree.dot") as  f:
	dot_graph = f.read()
display(graphviz.Source(dot_graph))

  # feature importance in tree
print(tree.feature_importances_) 

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)

plt.semilogy(x, y)  # Make a plot with log scaling on the y axis

# 4.2 decision tree regressor
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

# log transformation and transform back
  y1 = np.log(y)
  y2 = np.exp(y1)
  
  
# 4.3 RandomForest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
			
# 4.4 Gradient boosted regression trees (Gradient Boosting Machines)
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, 
                                   max_depth=1,
								   learning_rate = 0.01)

# 5. Support vector machines
from sklearn.svm import LinearSVC
svc = LinearSVC(C= , gamma = ).fit(x, y)  # C: regularization, gamma: how flexible the kernel is, over-fitting

# make a 3D figure
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
# plot first all the points with y==0, then all with y == 1
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature2")

# 6. neural network
np.tanh()         # tanh, similar as sigmoid
np.maximum(x, 0)  # relu

from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron classifier
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10]).fit(X_train, y_train)
# using two hidden layers, with 10 units each
mlp = MLPClassifier(solver='lbfgs', random_state=0,
                    hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
# using two hidden layers, with 10 units each, now with tanh nonlinearity.
mlp = MLPClassifier(solver='lbfgs', activation='tanh', alpha = 10,  # alpha: penalty or regularization
                    random_state=0, hidden_layer_sizes=[10, 10])
					
# stochastic 随机

# 7. Uncertainty estimates, decision function
# we rename the classes "blue" and "red" for illustration purposes:

gbrt.predict_proba(X_test)
gbrt.decision_function(X_test)

np.argmax(gbrt.decision_function(X_test), axis=1) # get the most possible class


y_named = np.array(["blue", "red"])[y]
np.all  # test if all elements in two arrays are identical



























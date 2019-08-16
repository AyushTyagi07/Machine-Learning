from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a, b)

import random
class ScrappyKNN():
	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closet(row)
			predictions.append(label)
		return predictions

	def closet(self, row):
		best_dis = euc(row, self.X_train[0])
		best_ind = 0
		for i in range(1,len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dis:
				best_dis=dist
				best_ind=i
		return self.Y_train[best_ind]


from sklearn import datasets
iris = datasets.load_iris()

X= iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .5)

from sklearn import tree
from sklearn.metrics import accuracy_score

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train,Y_train)
predictions = my_classifier.predict(X_test)

#from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

#from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = ScrappyKNN()

my_classifier2.fit(X_train,Y_train)
predictions = my_classifier2.predict(X_test)

print(accuracy_score(Y_test, predictions))
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
tes_idx = [0,50,100]

train_target = np.delete(iris.target, tes_idx)
train_data = np.delete(iris.data, tes_idx, axis=0) 

test_target = iris.target[tes_idx]
test_data = iris.data[tes_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
print(iris.target_names)
print(test_target)
print(clf.predict(test_data))

#viz

from six import StringIO
import pydotplus
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(clf,
	out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True, rounded=True,
	impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
Image(graph.create_png())
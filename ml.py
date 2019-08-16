from sklearn import tree
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [1, 1, 0, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
w = int(input("Enter fruit's Weight :"))
f = input("Select the fruit's Texture : \n 0) Bumpy \n1) Smooth\n")
op = clf.predict([[w, f]])
if op==0:
	print("Orange")
else:
	print("Apple")

# coding: utf-8
# created at 2020/4/10
__author__ = "fripSide"

from math import log
import numpy as np
import pandas as pd

# Simple Decision Tree
"""
windows需要安装 graphviz

First Pass: Simple Decision Tree (C4.5)


Pass 2: Time-Changing Decision Tree
timestamp, hierarchical features, normal features, access results, and other non-related fields. Table
"""


class DecisionNode:

	def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
		pass


class BaseDecisionTree:

	def build_tree(self, dataset, labels):
		pass

	def classify(self, test):
		pass

	def dump_tree_pdf(self):
		pass


class C45DecisionTree(BaseDecisionTree):

	def __init__(self):
		self.labels = []
		self.tree = None

	def build_tree(self, dataset, labels):
		self.labels = labels.copy()
		self.tree = self.create_tree(dataset, labels)

	def split_dateset(self, dataset, axis, value):
		"""按照给定特征划分数据集；去除选择维度(axis)中等于value的项"""
		ret_data = []
		for feat_vec in dataset:
			if feat_vec[axis] == value:
				reduce_feat = feat_vec[:axis]
				reduce_feat.extend(feat_vec[axis + 1:])
				ret_data.append(reduce_feat)
		return ret_data

	def _calc_shannon_ent(self, dataset):
		# H = -sigma(1, n) * P(xi) * log2(P(xi))
		num_entries = len(dataset)
		label_counts = {}
		for feat_vec in dataset:
			label = feat_vec[-1]
			if label not in label_counts:
				label_counts[label] = 0
			label_counts[label] += 1
		entropy = 0.0
		for key in label_counts:
			px = float(label_counts[key]) / num_entries
			entropy -= px * log(px, 2)
		return entropy

	def choose_best_split_feature(self, dataset):
		"""选择熵最大的特征作为分隔点"""
		num_features = len(dataset[0]) - 1
		base_entropy = self._calc_shannon_ent(dataset)
		best_gain_radio = 0.0
		best_feature = -1
		for i in range(num_features):
			feat_list = [exp[i] for exp in dataset]
			unique_vals = set(feat_list)
			new_entropy = 0.0
			split_info = 0.0
			for val in unique_vals:
				sub_dataset = self.split_dateset(dataset, i, val)
				prob = len(sub_dataset) / float(len(dataset))
				new_entropy += prob * self._calc_shannon_ent(sub_dataset)
				split_info += -prob * log(prob, 2)
			info_gain = base_entropy - new_entropy
			if split_info == 0:
				continue
			info_gain_radio = info_gain / split_info
			if info_gain_radio > best_gain_radio:
				best_gain_radio = info_gain_radio
				best_feature = i
		return best_feature

	def find_majority(self, cls_list):
		import operator
		cls_cnt = {}
		for vote in cls_list:
			cnt = cls_cnt.get(vote, 0)
			cnt += 1
			cls_cnt[vote] = cnt
		sorted_cnt = sorted(cls_cnt.items(), key=operator.itemgetter(1), reverse=True)
		return sorted_cnt[0][0]

	def create_tree(self, data_set, labels):
		cls_list = [exp[-1] for exp in data_set]
		if cls_list.count(cls_list[0]) == len(cls_list):
			return cls_list[0]
		if len(data_set[0]) == 1:
			return self.find_majority(cls_list)

		best_feat = self.choose_best_split_feature(data_set)
		best_feat_label = labels[best_feat]
		my_tree = {best_feat_label: {}}
		del labels[best_feat]
		feat_values = [exp[best_feat] for exp in data_set]
		unique_vals = set(feat_values)
		for val in unique_vals:
			sub_labels = labels[:]
			sub_dataset = self.split_dateset(data_set, best_feat, val)
			my_tree[best_feat_label][val] = self.create_tree(sub_dataset, sub_labels)
		return my_tree

	def _classify_one(self, input_tree, feat_labels, test_vec):
		first_str = list(input_tree.keys())[0]
		second_dict = input_tree[first_str]
		feat_index = feat_labels.index(first_str)

		for key in second_dict:
			if test_vec[feat_index] == key:
				if isinstance(second_dict[key], dict):
					cls_label = self._classify_one(second_dict[key], feat_labels, test_vec)
				else:
					cls_label = second_dict[key]
		return cls_label

	def classify(self, test):
		cls_labels = []
		for item in test:
			res = self._classify_one(self.tree, self.labels, item)
			cls_labels.append(res)
		return cls_labels

	def dump_tree_pdf(self):
		print("Tree", self.tree)
		import graphviz
		dot = graphviz.Digraph()

		def add_level(nodes, dot, pt):
			for key in nodes:
				dot.node(str(key))
				if pt:
					dot.edge(str(pt), str(key))
				if isinstance(nodes[key], dict):
					add_level(nodes[key], dot, key)

		add_level(self.tree, dot, None)
		dot.format = 'pdf'
		dot.render("my_tree")


class SkLearnDecisionTree(BaseDecisionTree):

	def __init__(self):
		self.clf = None
		self.X = []
		self.y = []
		self.labels = []
		self.feature_names = []
		self.target_names = []
		self.data_encoding = {}  # reverse data from number to label

	def build_tree(self, dataset, labels):
		from sklearn import tree
		self.labels = labels.copy()
		self._process_data(dataset, labels)
		clf = tree.DecisionTreeClassifier(criterion='entropy')
		self.clf = clf.fit(self.X, self.y)

	def classify(self, test):
		import warnings
		warnings.filterwarnings(action='ignore', category=DeprecationWarning)
		df = pd.DataFrame(test)
		df_new = pd.DataFrame()
		for col in df:
			ft = df[col].values
			if df[col].dtype == np.object:
				label = self.labels[col]
				le = self.data_encoding[label]
				if le:
					ft = le.transform(df[col].values)
			df_new[col] = ft
		res = self.clf.predict(df_new)
		res_le = self.data_encoding[self.labels[-1]]
		return res_le.inverse_transform(res)

	def _process_data(self, dataset, labels):
		"""将数据encoding，标签映射成数字
		https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
		"""
		from sklearn.preprocessing import LabelEncoder
		df = pd.DataFrame(dataset)
		df.apply(LabelEncoder().fit_transform)
		df_new = pd.DataFrame()
		for col in df:
			le = None
			ft = df[col].values
			if df[col].dtype == np.object:
				label_encoder = LabelEncoder()
				le = label_encoder.fit(df[col].unique())
				ft = le.transform(df[col].values)
			# iv = le.inverse_transform(ft)
			label = labels[col]
			self.data_encoding[label] = le
			df_new[col] = ft

		self.X = df_new.iloc[:, :-1]
		self.y = df_new.iloc[:, -1]

		self.feature_names = labels[:-1]
		self.target_names = df.iloc[:, -1].unique()

	# print(self.feature_names, self.target_names)

	def dump_tree_pdf(self):
		from sklearn import tree
		from sklearn.externals.six import StringIO
		import pydotplus
		pdf_data = StringIO()
		tree.export_graphviz(self.clf, out_file=pdf_data, feature_names=self.feature_names,
							 class_names=self.target_names,
							 filled=True, rounded=True, special_characters=True)
		graph = pydotplus.graph_from_dot_data(pdf_data.getvalue())
		graph.write_pdf("sklearn.pdf")


class TimeChangeDecisionTree:
	pass


def sklearn_test():
	from sklearn import tree
	from sklearn.datasets import load_iris
	from sklearn.externals.six import StringIO
	import pydotplus
	iris = load_iris()
	X = iris.data
	y = iris.target
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	clf = clf.fit(X, y)
	print(X, y)
	print(iris.feature_names, iris.target_names)

	pdf_data = StringIO()
	dot_data = tree.export_graphviz(clf, out_file=pdf_data, feature_names=iris.feature_names,
									class_names=iris.target_names,
									filled=True, rounded=True, special_characters=True)

	graph = pydotplus.graph_from_dot_data(pdf_data.getvalue())
	graph.write_pdf("iris.pdf")


def get_current_data():
	import dataset
	# 随意的测试数据
	data, label, test = dataset.create_test_data()
	truth = test[-1]
	test = test[:-1]

	# pdiff 数据
	data, label, test, truth = dataset.load_pdiff_data()
	return data, label, test, truth


def detect_changes(test, prd, truth):
	for idx, item in enumerate(test):
		if prd[idx] != truth[idx]:
			print("Policy Changed: ", item, prd[idx], "->", truth[idx])


def sklearn_results(data, label, test, truth):
	sk_tree = SkLearnDecisionTree()
	sk_tree.build_tree(data.copy(), label.copy())
	sk_tree.dump_tree_pdf()
	res = sk_tree.classify(test.copy())
	for idx, item in enumerate(test):
		print(item, "--->", res[idx])
	detect_changes(test, res, truth)


def my_tree_results(data, label, test, truth):
	tree = C45DecisionTree()
	tree.build_tree(data.copy(), label.copy())
	tree.dump_tree_pdf()
	res = tree.classify(test.copy())
	for idx, item in enumerate(test):
		print(item, "--->", res[idx])
	detect_changes(test, res, truth)

def main():
	data, label, test, truth = get_current_data()
	print("my C4.5 decision tree results:")
	my_tree_results(data, label, test, truth)
	print("\n------------------------\n")
	print("sklearn resutls:")
	sklearn_results(data, label, test, truth)
	print("\n------------------------\n")
	print("Current is:", truth)


if __name__ == "__main__":
	main()

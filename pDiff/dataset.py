# coding: utf-8
# created at 2020/4/10
__author__ = "fripSide"

"""
默认： train_data格式
每一行：属性 + 结果
"""

def create_test_data():
	train_data = [['slashdot', 'USA', 'yes', 18, 'None'],
				  ['google', 'France', 'yes', 23, 'Premium'],
				  ['digg', 'USA', 'yes', 24, 'Basic'],
				  ['kiwitobes', 'France', 'yes', 23, 'Basic'],
				  ['google', 'UK', 'no', 21, 'Premium'],
				  ['(direct)', 'New Zealand', 'no', 12, 'None'],
				  ['(direct)', 'UK', 'no', 21, 'Basic'],
				  ['google', 'USA', 'no', 24, 'Premium'],
				  ['slashdot', 'France', 'yes', 19, 'None'],
				  ['digg', 'USA', 'no', 18, 'None'],
				  ['google', 'UK', 'no', 18, 'None'],
				  ['kiwitobes', 'UK', 'no', 19, 'None'],
				  ['digg', 'New Zealand', 'yes', 12, 'Basic'],
				  ['slashdot', 'UK', 'no', 21, 'None'],
				  ['google', 'UK', 'yes', 18, 'Basic'],
				  ['kiwitobes', 'France', 'yes', 19, 'Basic']]
	labels = ["web", "country", "use", "age", "level"]
	test_data = [['slashdot', 'UK', 'no', 21], ['(direct)', 'UK', 'no', 21],
				 ['kiwitobes', 'France', 'yes', 19], ['google', 'USA', 'no', 24],
				 ['None', 'Basic', 'Basic', 'Premium']]  # truth

	return train_data, labels, test_data


def load_csv_as_list(filename):
	import pandas as pd
	df = pd.read_csv(filename, header=None)
	ret = []
	for row in df.iterrows():
		data = row[1]
		ret.append(list(data.values))
	return ret


def load_pdiff_data():
	train_data = load_csv_as_list("data.csv")
	labels = ["action", "url", "result"]
	df_test = load_csv_as_list("change.csv")
	test_data = []
	truth = []
	for item in df_test:
		test_data.append(item[:-1])
		truth.append(item[-1])
	return train_data, labels, test_data, truth


def main():
	df = load_csv_as_list("data.csv")
	print(df)
	pass


if __name__ == "__main__":
	main()

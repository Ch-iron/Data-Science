import sys
import numpy as np
import pandas as pd

training_set = sys.argv[1]
test_set = sys.argv[2]
result = sys.argv[3]

# information gain 구하기
    # entropy구하기
def info(data):
    info_data = 0
    for classi in data.groupby(list(data)[-1]):
        p = len(classi[1]) / len(data)
        info_data = info_data - p * np.log2(p)
    return info_data

    # infoA 구하기
def info_attr(data, attr):
    info_attr_data = 0
    for classj in data.groupby(attr):
        info_attr_data = info_attr_data + len(classj[1]) / len(data) * info(classj[1])
    return info_attr_data

    # gain 구하기
def get_gain(data, attr):
    gain = info(data) - info_attr(data, attr)
    return gain

    # splitinfoaD 구하기
def splitinfo_attr(data, attr):
    splitinfo_attr_data = 0
    for classi in data.groupby(attr):
        p = len(classi[1]) / len(data)
        splitinfo_attr_data = splitinfo_attr_data - p * np.log2(p)
    return splitinfo_attr_data

    # gain_ratio 구하기
def get_gain_ratio(data, attr):
    split = splitinfo_attr(data, attr)
    if split != 0:
        gain_ratio = get_gain(data, attr) / split
    else:
        gain_ratio = 0
    return gain_ratio

    # test attribute 구하기 - gain으로
def get_test_attribute_gain(data):
    max_gain = 0
    max_attr = 'end'
    for attr in data.iloc[:, 0: -1]:
        if max_gain < get_gain(data, attr):
            max_gain = get_gain(data, attr)
            max_attr = attr
    return max_attr

    # test attribute 구하기 - gain_ratio로
def get_test_attribute_gain_ratio(data):
    max_gain_ratio = 0
    max_attr = 'end'
    for attr in data.iloc[:, 0: -1]:
        if max_gain_ratio < get_gain_ratio(data, attr):
            max_gain_ratio = get_gain_ratio(data, attr)
            max_attr = attr
    return max_attr

def train_gain(data, first_attr):
    split = data.groupby(first_attr)
    for i in split:
        attr = get_test_attribute_gain(i[1])
        node = [first_attr, i[0], attr]
        if attr != 'end':
            test_attr.append(node)
            train_gain(i[1], attr)
        else:
            node[2] = i[1][list(training_data)[-1]].unique()[0]
            test_attr.append(node)

def train_gain_ratio(data, first_attr):
    split = data.groupby(first_attr)
    for i in split:
        attr = get_test_attribute_gain_ratio(i[1])
        node = [first_attr, i[0], attr]
        if attr != 'end':
            test_attr.append(node)
            train_gain_ratio(i[1], attr)
        else:
            node[2] = i[1][list(training_data)[-1]].unique()[0]
            test_attr.append(node)

def test(data, first_attr):
    for i in range(0, len(data)):
        split_attr = first_attr
        for j in test_attr:
            if split_attr in label:
                data.loc[i][list(training_data)[-1]] = split_attr
                break
            if j[0] == split_attr and data.iloc[i][split_attr] == j[1]:
                split_attr = j[2]


# 트레이닝 셋 불러오기
training_data = pd.read_csv('./' + training_set, sep='\t', engine='python', encoding='cp949')
label = training_data[list(training_data)[-1]].unique()

# first test attribute
test_attr = []
first_test_attr = ['root', 'root', get_test_attribute_gain(training_data)]
test_attr.append(first_test_attr)

#### training
train_gain(training_data, test_attr[0][2])
# train_gain_ratio(training_data, test_attr[0][2])

#### 테스트 셋 불러오기
testing_data = pd.read_csv('./' + test_set, sep='\t', engine='python', encoding='cp949')
testing_data[list(training_data)[-1]] = ''

#### test
test(testing_data, test_attr[0][2])

testing_data.to_csv(result, sep='\t', index=False)
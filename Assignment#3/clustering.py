import sys
import numpy as np
import pandas as pd
import random

sys.setrecursionlimit(10**4)

input = sys.argv[1]
n = int(sys.argv[2])
eps = int(sys.argv[3])
minpts = int(sys.argv[4])

cluster_cnt = 0

def distance(point1, point2):
    return np.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# 인풋 데이터 불러오기
input_data_pd = pd.read_csv('./' + input, sep='\t',  names=['index', 'x', 'y'], engine='python', encoding='cp949')
input_data = input_data_pd.values.tolist()

core_points = []
cluster = []

## 첫 포인트 잡기
def find_first_point_cluster():
    global core_points, cluster

    while True:
        core_point = input_data[random.randrange(0, len(input_data))]
        point_cluster = []
        for point in input_data:
            dist = distance(core_point, point)
            if dist <= eps:
                point_cluster.append(point)
        if(len(point_cluster) >= minpts):
            core_points.append(core_point)
            cluster.append(point_cluster)
            cluster = sum(cluster, [])
            break
    return point_cluster

## 클러스터에 속한 모든 점에서의 클러스터들을 구한다.
## 체인형식으로 들어가므로 재귀를 사용
def find_cluster(prev_point_cluster):
    global cluster, core_points

    for point in prev_point_cluster:
        if point in core_points:
            continue
        core_point = point
        point_cluster = []
        for point in input_data:
            dist = distance(core_point, point)
            if dist <= eps:
                point_cluster.append(point)
        if(len(point_cluster) >= minpts):
            core_points.append(core_point)
            cluster = cluster + point_cluster
            find_cluster(point_cluster)

## 클러스터 중복제거하여 완성하고 기존 데이터에서 클러스터의 데이터 제거
while cluster_cnt < n:
    point_cluster = find_first_point_cluster()
    find_cluster(point_cluster)
    tmp = pd.DataFrame(cluster, columns=['index', 'x', 'y'])
    tmp.drop_duplicates(['index'], inplace=True, ignore_index=True)
    for i in range(0, len(tmp)):
        idx = input_data_pd[input_data_pd['index'] == int(tmp.loc[i, ['index']])].index
        input_data_pd.drop(idx, axis=0, inplace=True)
    tmp.drop(['x', 'y'], axis=1, inplace=True)
    tmp['index'] = tmp['index'].astype(int)
    tmp.to_csv(input[:6] + '_cluster_' + str(cluster_cnt) + '.txt', index=False, header=False)
    cluster_cnt += 1
    cluster = []
    input_data = input_data_pd.values.tolist()
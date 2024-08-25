import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

training_set = sys.argv[1]
test_set = sys.argv[2]
file_name = training_set.split('.')[0]

# 트레이닝 셋 불러오기
training_data = pd.read_csv('./' + training_set, sep='\t', names=['user', 'movie', 'rating', 'time'], engine='python', encoding='cp949')
test_data = pd.read_csv('./' + test_set, sep='\t', names=['user', 'movie', 'rating', 'time'], engine='python', encoding='cp949')

# 트레이닝 셋과 테스트 셋 중에서 더 큰 값으로 매트릭스 사이즈 세팅
user_max = np.max(np.array([training_data.loc[len(training_data) - 1, 'user']], test_data.loc[len(test_data) - 1, 'user']))
movie_max = np.max(np.array([training_data.loc[training_data['movie'].argmax(), 'movie'], test_data.loc[test_data['movie'].argmax(), 'movie']]))

# rating_matrix 생성하기(nan값은 모두 0으로)
rating_matrix = np.zeros((user_max, movie_max), dtype=float)
for i in range(0, len(training_data)):
    rating_matrix[training_data.loc[i, 'user'] - 1, training_data.loc[i, 'movie'] - 1] = training_data.loc[i, 'rating']

# zero injection안쓰므로 사용안함
# pre_use_preference 매트릭스 생성하기(nan값은 모두 0으로)
# pre_use_preference = np.zeros((user_max, movie_max), dtype=float)
# for i in range(0, user_max):
#     for j in range(0, movie_max):
#         if rating_matrix[i, j] > 0:
#             pre_use_preference[i, j] = 1

# 오차 구하는 함수
def get_rmse(R, P, Q, imputed):
    full_pred_matrix = P @ Q.T

    x_impute_idx = []
    y_impute_idx = []

    for i in imputed:
        x_impute_idx.append(i[0])
    for i in imputed:
        y_impute_idx.append(i[1])

    R_impute = R[x_impute_idx, y_impute_idx]
    full_pred_matrix_imputed = full_pred_matrix[x_impute_idx, y_impute_idx]

    rmse = np.sqrt(mean_squared_error(R_impute, full_pred_matrix_imputed))
     
    return rmse

# sgd알고리즘으로 행렬분해한 각각의 feature행렬들 구하고 행렬 복원하는 함수
def sgd(epochs, lr, lmbd, imputed, origin):
    P = np.random.normal(scale=1 / k, size=(user_max, k))
    Q = np.random.normal(scale=1 / k, size=(movie_max, k))

    for epoch in range(0, epochs):
        for i, j, r in imputed:
            err = r - np.dot(P[i, :], Q[j, :].T)

            P[i, :] = P[i, :] + lr * (err * Q[j, :] - lmbd * P[i, :])
            Q[j, :] = Q[j, :] + lr * (err * P[i, :] - lmbd * Q[j, :])
    
        # rmse = get_rmse(origin, P, Q, imputed)
        if (epoch % 50) == 0 :
            print("epoch: ", epoch)

    return P @ Q.T

k = 2
epochs = 350
lr = 0.01
lmbd = 0.01

##### zero injection 사용안함
# pre_use_preference 매트릭스에서 값이 있는 것만 뽑아내고 sgd를 돌려서 0이었던 곳에 값 채워넣기
# pre_use_high = []
# for i in range(0, user_max):
#     for j in range(0, movie_max):
#         if pre_use_preference[i, j] > 0:
#             pre_use_high.append((i, j, pre_use_preference[i, j]))

# pre_use_preference = sgd(epochs, lr, lmbd, pre_use_high, pre_use_preference)

# pre_use_preference의 값이 0.9 초과이고 rating_matrix에서 값이 없던 곳은 nan이라는 의미로 6을 부여
# pre_use_preference의 값이 0.9 이하이고 rating_matrix에서 값이 없던 곳은 가장 낮은 점수인 1을 부여
# for i in range(0, user_max):
#     for j in range(0, movie_max):
#         if pre_use_preference[i, j] > 0.9 and rating_matrix[i, j] == 0:
#             rating_matrix[i, j] = 6
#         elif pre_use_preference[i, j] <= 0.9 and rating_matrix[i, j] == 0:
#             rating_matrix[i, j] = 1

# rating_matrix에서 값이 0 초과인 것만 뽑아내고 sgd를 돌려서 0이었던 곳에 값 채워넣기
ratings = []
for i in range(0, user_max):
    for j in range(0, movie_max):
        if rating_matrix[i, j] > 0:
            ratings.append((i, j, rating_matrix[i, j]))

# sgd로 rating_matrix학습
rating_matrix = sgd(epochs, lr, lmbd, ratings, rating_matrix)

# 테스트 셋 불러와서 rating, time값 없애고 예측한 값 집어넣어서 결과 파일 생성하기
prediction_data = test_data.drop(['rating', 'time'], axis=1)
for i in range(0, len(prediction_data)):
    prediction_data.loc[i, 'rating'] = rating_matrix[prediction_data.loc[i, 'user'] - 1, prediction_data.loc[i, 'movie'] - 1]

prediction_data.to_csv(file_name + '.base_prediction.txt', sep='\t', header=False, index=False) 
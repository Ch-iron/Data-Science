import sys
from itertools import combinations

minimum_support = sys.argv[1]
inputFile = sys.argv[2]
outputFile = sys.argv[3]

input = open(inputFile, 'r')

# 데이터 리스트화하기
data = input.readlines()
transactions = []
for i in data:
    transaction = i.strip('\n')
    transaction = transaction.split('\t')
    transaction = list(map(int, transaction))
    transactions.append(transaction)
subset_per_tx = []
for tx in transactions:
    tmp = []
    for i in range(len(tx) + 1):
        tmp.append(list(combinations(tx, i)))
    tmp = sum(tmp, [])
    subset_per_tx.append(tmp[1:])

# 전체 트랜잭션 개수
alltx = len(transactions)

# 최소 서포트의 확률을 개수로 환산
minimum = int(minimum_support) * alltx / 100

## 원소가 1개인 frequent pattern찾기
# 각 원소가 몇 개씩 있는지 카운트
counter = {}
for tx in transactions:
    for data in tx:
        try: counter[data] += 1
        except: counter[data] = 1

# 미니멈 서포트 개수보다 적지 않은 것 골라내기
counter = { key: value for key, value in counter.items() if value >= minimum }
frequent_items = dict(sorted(counter.items()))

# L1: 원소의 개수가 1개인 frequent pattern
L1 = []
for data in frequent_items:
    L1.append(int(data))

# 출력할 모든 frequent pattern(2개 이상)을 담을 리스트
allcounter = {}
prevL = []
k = 2

## 원소가 k(k >= 2)개인 frequent pattern찾기 
while True:
# 원소가 k개인 후보를 L에서 만들기
    counter = {}
    if k == 2:
        C = list(combinations(L1, k))
    else:
        C = list(combinations(L, k))
    # 후보들이 각 트랜잭션에 속해있는지를 판별하고 개수를 카운트한다.
    for index, tx in enumerate(transactions):
        tmp = []
        if len(tx) >= k:
            for subset in subset_per_tx[index]:
                if len(subset) == k:
                    tmp.append(subset)
            for i in tmp:
                for j in C:
                    if set(i) == set(j):
                        try: counter[j] += 1
                        except: counter[j] = 1
    # 개수가 미니멈 서포트보다 적지 않은 것만 골라낸다.
    counter = { key: value for key, value in counter.items() if value >= minimum }
    frequent_items = dict(sorted(counter.items()))
    allcounter.update(frequent_items)
    # frequent pattern이 0개라면 종료
    if len(frequent_items) == 0:
        break
    
    print('Finding L' + str(k) + ' frequent pattern finish')

    # 원소가 2개이상인 freqent_pattern: L
    L = []
    for data in frequent_items:
        L.append(list(data))

    # 원소가 3개 이상인 것 중에서 서브셋이 frequentset이 아닌것 골라내기
    if k > 2:
        for fp in L:
            tmp = list(combinations(fp, k - 1))
            for i in tmp:
                p = 0
                for prevfp in prevL:
                    if set(prevfp) == set(i):
                        p = p + 1
                        break
                if p == 0:
                    L.remove(fp)
    prevL = L

    # L을 셀프조인하기 위해 unique한 값들로 풀어헤침
    L = sum(L, [])
    Lset = set(L)
    L = list(Lset)

    # 개수 1씩 증가
    k = k + 1

## 찾은 패턴의 서포트를 개수가 아닌 확률로 변경하기
for key, value in allcounter.items():
    allcounter[key] = value / alltx * 100
print('Calculating each element\'s support finished. Please wait...')

## 각 항목마다 association rule을 찾고 confidence를 구하여 output파일에 쓰기
output = open(outputFile, 'a')
for key, value in allcounter.items():
    subset = []
    for i in range(len(key) + 1):
        subset.append(list(combinations(key, i)))
    subset = sum(subset, [])
    subset = subset[1:]
    as_rule = []
    for i in subset:
        for j in subset:
            if set(i + j) == set(key) and len(i + j) == len(key):
                as_rule.append([i, j])
    for i in as_rule:
        confidence_denominator = 0
        confidence_numerator = 0
        for tx in subset_per_tx:
            for subset in tx:
                if set(i[0]) == set(subset):
                    confidence_denominator = confidence_denominator + 1
                    for subset2 in tx:
                        if set(i[1]) == set(subset2):
                            confidence_numerator = confidence_numerator + 1
                            break
                    break
        confidence = round(confidence_numerator / confidence_denominator * 100, 2)
        output.write('{')
        for left in i[0]:
            if left == i[0][len(i[0]) - 1]:
                output.write(str(left) + '}\t')
            else:
                output.write(str(left) + ',')
        output.write('{')
        for right in i[1]:
            if right == i[1][len(i[1]) - 1]:
                output.write(str(right) + '}\t')
            else:
                output.write(str(right) + ',')
        output.write(str(format(value, '.2f')) + '\t' + str(format(confidence, '.2f')) + '\n')
output.close()
print('Finding each element\'s association rule & calculating confidence finished')
import pandas as pd
from neighborhood_based.collaborative_filtering import CF
from numpy import sqrt

PATH_FILE = "data/formatted/"
r_cols = ["user_id", "item_id", "access_counting"]

## Bo comment cai nay de chay voi ca bo du lieu: industry type (it)
# ratings_base = pd.read_csv(PATH_FILE + "train_file_it.csv", sep=",", names=r_cols, encoding="latin-1")
# ratings_test = pd.read_csv(PATH_FILE + "test_file_it.csv", sep=",", names=r_cols, encoding="latin-1")

## Bo comment cai nay de chay voi ca bo du lieu : advertiser ID (ai)
ratings_base = pd.read_csv(PATH_FILE + "train_file_ai.csv", sep=",", names=r_cols, encoding="latin-1")
ratings_test = pd.read_csv(PATH_FILE + "test_file_ai.csv", sep=",", names=r_cols, encoding="latin-1")

rate_train = ratings_base.values
rate_test = ratings_test.values

# indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1

# user-user CF
rs = CF(rate_train, k = 30, uuCF = 1)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2

RMSE = sqrt(SE/n_tests)
rs.print_recommendation()
print('User-user CF, RMSE =', RMSE)

# item-item CF
rs2 = CF(rate_train, k = 30, uuCF = 0)
rs2.fit()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs2.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2

RMSE = sqrt(SE/n_tests)
rs.print_recommendation()
print('Item-item CF, RMSE =', RMSE)










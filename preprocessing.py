import pandas as pd
import numpy as np
from random import uniform
from collections import Counter

MODEL_BASED = "industry_type"
DATA_PATH = "data/formatted/"

raw_path_file_ad_log = "data/raw/ad_log.tsv.0063_part_00.gz"
raw_path_file_campaign = "data/raw/campaign_data_hust_2018_autumn_eng.csv"
raw_path_file_url = "data/raw/url_category_hust_2018_autumn.csv"

F_PATH_FILE_AD_LOG = "data/formatted/ad_log.csv"
F_PATH_FILE_CAMPAIGN = "data/formatted/campaign_2018_autumn.csv"

ad_log_header = ["view_datetime", "user_id", "purl_subdomain", "purl_base", "os", "device", "source_log_nm",
              "advertiser_id", "campaign_id", "flight_id", "ip_address"]

#### 1. Handle biggest data file
# # Handle ad_log_file, read whole file but separate chunks
# chunksize = 100000
# df_iter = pd.read_csv(raw_path_file_ad_log, delimiter = "\t", chunksize=chunksize, iterator=True)
# for iter_num, chunk in enumerate(df_iter, 0):
#     print("Processing chunk: ", iter_num)
#     saved = chunk[chunk["advertiser_id"]!= 0]
#     saved.to_csv(F_PATH_FILE_AD_LOG, header=None, sep=',', mode='a', index=False, encoding='utf-8')


## Handle campaign_file
# df_cp = pd.read_csv(raw_path_file_campaign, header=0, delimiter = ',', error_bad_lines=False)
# # ,No.,FlightID,advertiser_id,campaign_id,Campaign Name,キャンペーン名,Line Item Name,ラインアイテム名,Type of industry,業種,RTB) campaign merchandise,RTB）キャンペーン商材,OTrans,RTB）キャンペーン商材（詳細）
# cp_header = ["flight_id", "advertiser_id", "campaign_id", "campaign_name_eng", "campaign_name_jp",
#               "viewer_type_eng", "viewer_type_jp", "industry_type_eng", "industry_type_jp",
#               "rtb_campaign_material_eng", "rtb_campaign_material_jp", "campaign_product_eng", "campaign_product_details_jp"]
# df_cp = df_cp.iloc[:, 2:]
# df_cp.to_csv(F_PATH_FILE_CAMPAIGN, header=cp_header, sep=',', index=False, encoding='utf-8')


#### 2. Join tables
df1 = pd.read_csv(F_PATH_FILE_AD_LOG, delimiter = ',', names=ad_log_header, error_bad_lines=False)
df2 = pd.read_csv(F_PATH_FILE_CAMPAIGN, delimiter = ',', header=0, error_bad_lines=False)

df1["flight_id"] = pd.to_numeric(df1["flight_id"], errors="coerce")     # Error when merging files
df1 = df1[df1["flight_id"] != None]

df_join = df1.join(df2[['flight_id', 'industry_type_eng']].set_index('flight_id'), on='flight_id')
df_join = df_join.reset_index()

if MODEL_BASED == "industry_type":
    df_join = df_join[df_join['industry_type_eng'].notnull()]
df = df_join.iloc[:, 1:]
df = df.reset_index(drop=True)
print(df.head(10))

#### 3. Handle datetime
def func_convert_time(a):
    if 0 <= a and a <=5:
        return 0    # Sleep
    elif 6 == a:
        return 1    # Breakfast
    elif 7 == a:
        return 2    # Commute
    elif 8 <= a and a <= 11:
        return 3    # Office hours
    elif 12 <= a and a <= 13:
        return 4    # Lunch
    elif 14 <= a and a <= 17:
        return 5    # Office hours
    elif a == 18:
        return 6    # Commute
    elif a == 19:
        return 7    # Dinner
    elif 20 <= a and a < 24:
        return 8    # Resting
    else:
        return -1

hours = pd.DatetimeIndex(df["view_datetime"]).hour
df_hour1 = pd.DataFrame(data={"view_hour": hours}, index=df.index, dtype=np.int64)
df_hour2 = pd.to_numeric(df_hour1["view_hour"].apply(func_convert_time))
df = pd.concat([df, df_hour2], axis=1)
print(df.tail(100))
print("Number of records: ", df.shape[0])

#### 4. Filter records which are appear at least n times by the same user
#df = df[df.duplicated(subset=["user_id"], keep=False)]                     # n >= 2
counts = Counter(df["user_id"])
df = df[df["user_id"].isin([key for key in counts if counts[key] > 5])]     # n > 5
print("Number of records after remove users: ", df.shape[0])


#### 5. Handle ID collumns
user_labels, unique_users = pd.factorize(df["user_id"])
print("Number of uniques user is %d" %len(unique_users))
print(unique_users[:5])

advertiser_id_labels, unique_advertiser_id = pd.factorize(df["advertiser_id"])
print("Number of uniques advertiser_id is %d" %len(unique_advertiser_id))
print(unique_advertiser_id[:5])

campaign_id_labels, unique_campaign_id = pd.factorize(df["campaign_id"])
print("Number of uniques campaign_id is %d" %len(unique_campaign_id))
print(unique_campaign_id[:5])

flight_id_labels, unique_flight_id = pd.factorize(df["flight_id"])
print("Number of uniques flight_id is %d" %len(unique_flight_id))
print(unique_flight_id[:5])



#### 6. Handle categorical collumns
subdomain_labels, uniques_subdomain = pd.factorize(df['purl_subdomain'])
print("Number of uniques purl_subdomain is %d" %len(uniques_subdomain))
print(uniques_subdomain[:5])

purl_base_labels, uniques_purl_base = pd.factorize(df['purl_base'])
print("Number of uniques purl_base is %d" %len(uniques_purl_base))
print(uniques_purl_base[:5])

os_labels, unique_os = pd.factorize(df['os'])
print("Number of unique os is %d" %len(unique_os))
print(unique_os)

device_labels, unique_device = pd.factorize(df['device'])
print("Number of unique device is %d" %len(unique_device))
print(unique_device)

source_log_labels, unique_source_log = pd.factorize(df['source_log_nm'])
print("Number of unique source_log is %d" %len(unique_source_log))
print(unique_source_log)

ip_address_labels, unique_ip_address = pd.factorize(df['ip_address'])
print("Number of unique ip_address is %d" %len(unique_ip_address))
print(unique_ip_address)

industry_type_labels, unique_industry_type = pd.factorize(df['industry_type_eng'])
print("Number of unique industry_type_eng is %d" %len(unique_industry_type))
print(unique_industry_type)

# Re-feature
df['user_id'] = user_labels
df['advertiser_id'] = advertiser_id_labels
df['campaign_id'] = campaign_id_labels
df['flight_id'] = flight_id_labels
df['purl_subdomain'] = subdomain_labels
df['purl_base'] = purl_base_labels
df['os'] = os_labels
df['device'] = device_labels
df['source_log_nm'] = source_log_labels
df['ip_address'] = ip_address_labels
df['industry_type_eng'] = industry_type_labels


df = df.sort_values(by=["user_id"]).reset_index(drop=True)                  # sort by user_id then reset index
print(len(df["user_id"].unique()))                                          # 445 unique users with number of appearance at least 6 times


#### 7. Building matrix based on industry_type and advertiser_id
len_user = len(df["user_id"].unique())
len_it = 9 * len(unique_industry_type)
len_ai = 9 * len(unique_advertiser_id)

mat_it = np.zeros((len_user, len_it))
mat_ai = np.zeros((len_user, len_ai))
for id, u in df.iterrows():
    user_pos = u["user_id"]
    view_hour = u["view_hour"]
    industry_type = u["industry_type_eng"]
    advertiser_id = u["advertiser_id"]
    item_pos1 = view_hour * len(unique_industry_type) + industry_type
    item_pos2 = view_hour * len(unique_advertiser_id) + advertiser_id
    mat_it[user_pos][item_pos1] += 1
    mat_ai[user_pos][item_pos2] += 1

print(mat_it)
print(mat_ai)

np.savetxt(DATA_PATH + "mat_it.csv", mat_it, fmt='%d', delimiter=',', newline='\n', comments='')
np.savetxt(DATA_PATH + "mat_ai.csv", mat_ai, fmt='%d', delimiter=',', newline='\n', comments='')

train_file_it = []
test_file_it = []
train_file_ai = []
test_file_ai = []

#### 8. Building file user_id, item_id, access_counting
for i in range(len_user):
    for j in range(len_it):
        if uniform(0, 1) < 0.7:
            train_file_it.append([i, j, mat_it[i][j]])
        else:
            test_file_it.append([i, j, mat_it[i][j]])

    for k in range(len_ai):
        if uniform(0, 1) < 0.7:
            train_file_ai.append([i, k, mat_ai[i][k]])
        else:
            test_file_ai.append([i, k, mat_ai[i][k]])

np.savetxt(DATA_PATH + "train_file_it.csv", np.array(train_file_it), fmt='%d', delimiter=',', newline='\n', comments='')
np.savetxt(DATA_PATH + "test_file_it.csv", np.array(test_file_it), fmt='%d', delimiter=',', newline='\n', comments='')
np.savetxt(DATA_PATH + "train_file_ai.csv", np.array(train_file_ai), fmt='%d', delimiter=',', newline='\n', comments='')
np.savetxt(DATA_PATH + "test_file_ai.csv", np.array(test_file_ai), fmt='%d', delimiter=',', newline='\n', comments='')



















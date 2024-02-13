#!/usr/bin/env python
# coding: utf-8

# # 영업 성공 여부 분류 경진대회

# ## 1. 데이터 확인

# ### 필수 라이브러리

# In[1221]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import scipy.stats 


# ### 데이터 셋 읽어오기

# In[1222]:


df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)


# In[1223]:


df_train.head() # 학습용 데이터 살펴보기


# In[1225]:


df_test.head() # 테스트 데이터 살펴보기


# ## 2. 데이터 전처리

# ## 2-1 Nan 처리

# In[1226]:


nan_in_1st_column = df_train['bant_submit'].isna().any()
print("첫 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_1st_column)

nan_in_1st_column_test = df_test['bant_submit'].isna().any()
print("첫 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_1st_column)


# In[1227]:


nan_in_2nd_column = df_train['customer_country'].isna().any()
print("두 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_2nd_column)

nan_in_2st_column_test = df_test['customer_country'].isna().any()
print("두 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_2st_column_test)


# ### 2-1-1 고객의 국적 결측치 처리
# 고객의 국적 결측치는 "Unknown"으로 처리

# In[1228]:


df_train['customer_country'] = df_train['customer_country'].fillna("Unknown")

nan_in_2nd_column = df_train['customer_country'].isna().any()
print("두 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_2nd_column)


# In[1229]:


nan_in_3rd_column = df_train['business_unit'].isna().any()
print("세 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_3rd_column)

nan_in_3rd_column_test = df_test['business_unit'].isna().any()
print("세 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_3rd_column_test)


# In[1230]:


nan_in_4th_column = df_train['com_reg_ver_win_rate'].isna().any()
print("네 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_4th_column)

nan_in_4th_column_test = df_test['com_reg_ver_win_rate'].isna().any()
print("네 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_4th_column_test)


# ### 2-1-2 Vertical Level 1, business unit (region) 결측치 처리
# 고객의 Business unit region기준 oppty비율 결측치는 평균값으로 처리

# In[1231]:


mean_com_reg_ver_win_rate = df_train['com_reg_ver_win_rate'].mean()
df_train['com_reg_ver_win_rate'] = df_train['com_reg_ver_win_rate'].fillna(mean_com_reg_ver_win_rate)

mean_com_reg_ver_win_rate_test = df_test['com_reg_ver_win_rate'].mean()
df_test['com_reg_ver_win_rate'] = df_test['com_reg_ver_win_rate'].fillna(mean_com_reg_ver_win_rate_test)

nan_in_4th_column = df_train['com_reg_ver_win_rate'].isna().any()
print("네 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_4th_column)
nan_in_4th_column_test = df_test['com_reg_ver_win_rate'].isna().any()
print("네 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_4th_column_test)


# In[1232]:


nan_in_5th_column = df_train['customer_idx'].isna().any()
print("다섯 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_5th_column)

nan_in_5th_column_test = df_test['customer_idx'].isna().any()
print("다섯 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_5th_column_test)


# In[1233]:


nan_in_6th_column = df_train['customer_type'].isna().any()
print("여섯 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_6th_column)

nan_in_6th_column_test = df_test['customer_type'].isna().any()
print("여섯 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_6th_column_test)


# ### 2-1-3 Customer_type 결측치 처리
# 고객의 유형은 "Unknown"으로 처리

# In[1234]:


df_train['customer_type'] = df_train['customer_type'].fillna("Unknown")
df_test['customer_type'] = df_test['customer_type'].fillna("Unknown")

nan_in_6th_column = df_train['customer_type'].isna().any()
print("여섯 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_6th_column)

nan_in_6th_column_test = df_test['customer_type'].isna().any()
print("여섯 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_6th_column_test)


# In[1235]:


nan_in_7th_column = df_train['enterprise'].isna().any()
print("일곱 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_7th_column)

nan_in_7th_column_test = df_test['enterprise'].isna().any()
print("일곱 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_7th_column_test)


# In[1236]:


nan_in_8th_column = df_train['historical_existing_cnt'].isna().any()
print("여덟 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_8th_column)

nan_in_8th_column_test = df_train['historical_existing_cnt'].isna().any()
print("여덟 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_8th_column_test)


# ### 2-1-4 영업 전환 횟수 결측치 처리
# 영업 전환 횟수는 0으로 처리

# In[1237]:


df_train['historical_existing_cnt'] = df_train['historical_existing_cnt'].fillna(0)
df_test['historical_existing_cnt'] = df_test['historical_existing_cnt'].fillna(0)

nan_in_8th_column = df_train['historical_existing_cnt'].isna().any()
print("여덟 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_8th_column)

nan_in_8th_column_test = df_train['historical_existing_cnt'].isna().any()
print("여덟 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_8th_column_test)


# In[1238]:


nan_in_9th_column = df_train['id_strategic_ver'].isna().any()
print("아홉 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_9th_column)

nan_in_9th_column_test = df_test['id_strategic_ver'].isna().any()
print("아홉 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_9th_column_test)


# ### 2-1-5 특정사업부 가중치 결측치 처리 (id)
# 사업부 가중치 결측치는 0으로 처리 (idit에서 id, it 값 중 하나라도 1의 값을 가지면 1로 표현)

# In[1239]:


df_train['id_strategic_ver'] = df_train['id_strategic_ver'].fillna(0)
df_test['id_strategic_ver'] = df_test['id_strategic_ver'].fillna(0)

nan_in_9th_column = df_train['id_strategic_ver'].isna().any()
print("아홉 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_9th_column)
nan_in_9th_column_test = df_test['id_strategic_ver'].isna().any()
print("아홉 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_9th_column_test)


# In[1240]:


nan_in_10th_column = df_train['it_strategic_ver'].isna().any()
print("열 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_10th_column)

nan_in_10th_column_test = df_test['it_strategic_ver'].isna().any()
print("열 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_10th_column_test)


# ### 2-1-6 특정사업부 가중치 결측치 처리 (it)
# 2-1-5와 마찬가지로 0으로 처리

# In[1241]:


df_train['it_strategic_ver'] = df_train['it_strategic_ver'].fillna(0)
df_test['it_strategic_ver'] = df_test['it_strategic_ver'].fillna(0)

nan_in_10th_column = df_train['it_strategic_ver'].isna().any()
print("열 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_10th_column)

nan_in_10th_column_test = df_test['it_strategic_ver'].isna().any()
print("열 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_10th_column_test)


# In[1242]:


nan_in_11th_column = df_train['idit_strategic_ver'].isna().any()
print("열 한번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_11th_column)

nan_in_11th_column_test = df_test['idit_strategic_ver'].isna().any()
print("열 한번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_11th_column_test)


# ### 2-1-7 특정사업부 가중치여부 결측치 처리 (itid)
# 2-1-5와 마찬가지로 0으로 처리

# In[1243]:


df_train['idit_strategic_ver'] = df_train['idit_strategic_ver'].fillna(0)
df_test['idit_strategic_ver'] = df_test['idit_strategic_ver'].fillna(0)

nan_in_11th_column = df_train['idit_strategic_ver'].isna().any()
print("열 한번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_11th_column)
nan_in_11th_column_test = df_test['idit_strategic_ver'].isna().any()
print("열 한번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_11th_column_test)


# In[1244]:


nan_in_12th_column = df_train['customer_job'].isna().any()
print("열 두번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_12th_column)
nan_in_12th_column_test = df_test['customer_job'].isna().any()
print("열 두번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_12th_column_test)


# ### 2-1-8 고객 직업군 결측치 처리
# 고객 직업군 결측치는 "Unknown"으로 처리

# In[1245]:


df_train['customer_job'] = df_train['customer_job'].fillna("Unknown")
df_test['customer_job'] = df_test['customer_job'].fillna("Unknown")

nan_in_12th_column_test = df_test['customer_job'].isna().any()
print("열 두번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_12th_column_test)
nan_in_12th_column_test = df_test['customer_job'].isna().any()
print("열 두번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_12th_column_test)


# In[1246]:


nan_in_13th_column = df_train['lead_desc_length'].isna().any()
print("열 세번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_13th_column)

nan_in_13th_column_test = df_test['lead_desc_length'].isna().any()
print("열 세번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_13th_column_test)


# In[1247]:


nan_in_14th_column = df_train['inquiry_type'].isna().any()
print("열 네번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_14th_column)

nan_in_14th_column_test = df_test['inquiry_type'].isna().any()
print("열 네번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_14th_column_test)


# ### 2-1-9 고객의 문의 유형
# 고객의 문의 유형 결측치는 "Unknown"으로 처리

# In[1248]:


df_train['inquiry_type'] = df_train['inquiry_type'].fillna("Unknown")
df_test['inquiry_type'] = df_test['inquiry_type'].fillna("Unknown")

nan_in_14th_column = df_train['inquiry_type'].isna().any()
print("열 네번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_14th_column)
nan_in_14th_column_test = df_test['inquiry_type'].isna().any()
print("열 네번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_14th_column_test)


# In[1249]:


nan_in_15th_column = df_train['product_category'].isna().any()
print("열 다섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_15th_column)

nan_in_15th_column_test = df_test['product_category'].isna().any()
print("열 다섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_15th_column_test)


# ### 2-1-10 고객의 요청 제품 카테고리
# 고객 요청 제품 카테고리 결측치는 "Unknown"으로 처리

# In[1250]:


df_train['product_category'] = df_train['product_category'].fillna("Unknown")
df_test['product_category'] = df_test['product_category'].fillna("Unknown")

nan_in_15th_column = df_train['product_category'].isna().any()
print("열 다섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_15th_column)

nan_in_15th_column_test = df_test['product_category'].isna().any()
print("열 다섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_15th_column_test)


# ### 2-1-11, 12 고객의 요청 제품 하위 카테고리, 제품 모델명
# 고객의 요청제품 하위 카테고리,제품 모델명 결측치 모두 "Unknown"으로 처리

# In[1251]:


df_train['product_subcategory'] = df_train['product_subcategory'].fillna("Unknown")
df_test['product_subcategory'] = df_test['product_subcategory'].fillna("Unknown")

nan_in_16th_column = df_train['product_subcategory'].isna().any()
print("열 여섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_16th_column)
nan_in_16th_column_test = df_test['product_subcategory'].isna().any()
print("열 여섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_16th_column_test)

df_train['product_modelname'] = df_train['product_modelname'].fillna("Unknown")
df_test['product_modelname'] = df_test['product_modelname'].fillna("Unknown")

nan_in_17th_column = df_train['product_modelname'].isna().any()
print("열 일곱번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_17th_column)
nan_in_17th_column_test = df_test['product_modelname'].isna().any()
print("열 일곱번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_16th_column_test)


# In[1252]:


nan_in_18th_column = df_train['customer_country.1'].isna().any()
print("열 여덟번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_18th_column)

nan_in_18th_column_test = df_test['customer_country.1'].isna().any()
print("열 여덟번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_18th_column_test)


# ### 2-1-13 담당 자서 법인명 기반의 지역 정보
# 고객의 지역 정보 결측치는 "Unknown"으로 처리

# In[1253]:


df_train['customer_country.1'] = df_train['customer_country.1'].fillna("Unknown")

nan_in_18th_column = df_train['customer_country.1'].isna().any()
print("열 여덟번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_18th_column)


# In[1254]:


nan_in_19th_column = df_train['customer_position'].isna().any()
print("열 아홉번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_19th_column)

nan_in_19th_column_test = df_test['customer_position'].isna().any()
print("열 아홉번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_19th_column_test)


# In[1255]:


nan_in_20th_column = df_train['response_corporate'].isna().any()
print("스무 번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_20th_column)

nan_in_20th_column = df_test['response_corporate'].isna().any()
print("스무 번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_20th_column)


# In[1256]:


nan_in_21th_column = df_train['expected_timeline'].isna().any()
print("스물 한번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_21th_column)

nan_in_21th_column_test = df_test['expected_timeline'].isna().any()
print("스물 한번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_21th_column_test)


# ### 2-1-14 고객의 요청한 처리 일정
# 고객의 요청한 처리 일정 결측치는 "Unknown"으로 처리

# In[1257]:


df_train['expected_timeline'] = df_train['expected_timeline'].fillna("Unknown")
df_test['expected_timeline'] = df_test['expected_timeline'].fillna("Unknown")

nan_in_21th_column = df_train['expected_timeline'].isna().any()
print("스물 한번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_21th_column)

nan_in_21th_column_test = df_test['expected_timeline'].isna().any()
print("스물 한번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_21th_column_test)


# In[1258]:


nan_in_22th_column = df_train['ver_cus'].isna().any()
print("스물 두번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_22th_column)

nan_in_22th_column_test = df_test['ver_cus'].isna().any()
print("스물 두번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_22th_column_test)


# In[1259]:


nan_in_23th_column = df_train['ver_pro'].isna().any()
print("스물 세번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_24th_column)

nan_in_23th_column_test = df_test['ver_pro'].isna().any()
print("스물 세번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_23th_column_test)


# In[1260]:


nan_in_24th_column = df_train['ver_win_rate_x'].isna().any()
print("스물 네번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_24th_column)

nan_in_24th_column_test = df_test['ver_win_rate_x'].isna().any()
print("스물 네번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_24th_column_test)


# ### 2-1-15 vertical 수 비율과 vertical 별 lead 수 대비 영업 전환 성공 비율
# business_area 따라 달라진다는 가정하에 0으로 결측치 우선 처리

# In[1261]:


df_train['ver_win_rate_x'] = df_train['ver_win_rate_x'].fillna(0)
df_test['ver_win_rate_x'] = df_test['ver_win_rate_x'].fillna(0)

nan_in_24th_column = df_train['ver_win_rate_x'].isna().any()
print("스물 네번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_24th_column)

nan_in_24th_column_test = df_test['ver_win_rate_x'].isna().any()
print("스물 네번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_24th_column_test)


# In[1262]:


nan_in_25th_column = df_train['ver_win_ratio_per_bu'].isna().any()
print("스물 다섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_25th_column)
nan_in_25th_column_test = df_test['ver_win_ratio_per_bu'].isna().any()
print("스물 다섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_25th_column_test)


# ### 2-1-16 ver_win_ration_per_bu
# 결측치 0으로 우선 처리

# In[1263]:


df_train['ver_win_ratio_per_bu'] = df_train['ver_win_ratio_per_bu'].fillna(0)
df_test['ver_win_ratio_per_bu'] = df_test['ver_win_ratio_per_bu'].fillna(0)

nan_in_25th_column = df_train['ver_win_ratio_per_bu'].isna().any()
print("스물 다섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_25th_column)
nan_in_25th_column_test = df_test['ver_win_ratio_per_bu'].isna().any()
print("스물 다섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_25th_column_test)


# In[1264]:


nan_in_26th_column = df_train['business_area'].isna().any()
print("스물 여섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_26th_column)
nan_in_26th_column_test = df_test['business_area'].isna().any()
print("스물 여섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_26th_column_test)


# ### 2-1-17 고객의 사업 영역
# 고객의 사업 영역 결측치 "Unknown"으로 처리

# In[1265]:


df_train['business_area'] = df_train['business_area'].fillna("Unknown")
df_test['business_area'] = df_test['business_area'].fillna("Unknown")

nan_in_26th_column = df_train['business_area'].isna().any()
print("스물 여섯번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_26th_column)
nan_in_26th_column_test = df_test['business_area'].isna().any()
print("스물 여섯번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_26th_column_test)


# In[1266]:


nan_in_27th_column = df_train['business_subarea'].isna().any()
print("스물 일곱번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_27th_column)

nan_in_27th_column_test = df_test['business_subarea'].isna().any()
print("스물 일곱번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_27th_column_test)


# ### 2-1-18 고객의 세부 사업 영역
# 고객 세부 사업 영역 결측치 "Unknown"으로 처리

# In[1267]:


df_train['business_subarea'] = df_train['business_subarea'].fillna("Unknown")
df_test['business_subarea'] = df_test['business_subarea'].fillna("Unknown")

nan_in_27th_column = df_train['business_subarea'].isna().any()
print("스물 일곱번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_27th_column)
nan_in_27th_column_test = df_test['business_subarea'].isna().any()
print("스물 일곱번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_27th_column_test)


# In[1268]:


nan_in_28th_column = df_train['lead_owner'].isna().any()
print("스물 여덟번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_28th_column)
nan_in_28th_column_test = df_test['lead_owner'].isna().any()
print("스물 여덟번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_28th_column_test)


# In[1269]:


nan_in_29th_column = df_train['is_converted'].isna().any()
print("스물 아홉번째 컬럼에서 NaN 값이 있는지 여부:", nan_in_29th_column)
nan_in_29th_column_test = df_test['is_converted'].isna().any()
print("스물 아홉번째 테스트 컬럼에서 NaN 값이 있는지 여부:", nan_in_29th_column_test)


# ## 2-2 레이블 인코딩

# In[1270]:


def label_encoding(series: pd.Series) -> pd.Series:
    return series.astype(str).astype('category').cat.codes

# 레이블 인코딩할 컬럼들 정의
label_columns = [
    "customer_country", "business_subarea", "business_area", "business_unit",
    "customer_type", "enterprise", "customer_job", "inquiry_type",
    "product_category", "product_subcategory", "product_modelname",
    "customer_country.1", "customer_position", "response_corporate",
    "expected_timeline"
]

for col in label_columns:
    df_train[col] = label_encoding(df_train[col])
    df_test[col] = label_encoding(df_test[col])

def count_false(df, column_name):
    zero_count = (df['is_converted'] == False).sum()
    return zero_count

# print("false의 개수:", count_false(df_train, 'is_converted'))

def encode_boolean(column):
    """
    True는 1, False는 0
    """
    encoded_column = column.astype(int)
    return encoded_column

df_train['is_converted'] = encode_boolean(df_train['is_converted'])

def count_zeros(df):
    zero_count = (df['is_converted'] == 0).sum()
    return zero_count

# # 모든 열을 숫자로 변환
# def encode(df, columm_name):
#     if df[column].dtype == 'object':
#         df[column] = pd.Categorical(df[column])
#         df[column] = df[column].cat.codes
#     elif df[column].dtype == 'bool':
#         df[column] = df[column].astype(int)
#     elif df[column].dtype == 'datetime64[ns]':
#         df[column] = df[column].astype(int)
        
# for column in label_columns:
#     df_train[column] = encode(df_train, column)
#     df_test[column] = encode(df_test, column)
    
# print(df_train)
# print("0 개수:", count_zeros(df_train))


# 다시 학습 데이터와 제출 데이터를 분리합니다.

# In[1210]:


# for col in label_columns:  
#     df_train[col] = df_all.iloc[: len(df_train)][col]
#     df_test[col] = df_all.iloc[len(df_train) :][col]


# ## 2-3 Outlier 확인

# In[1271]:


# Outlier check (we will not drop outlier)
plt.figure(figsize=(20, 6))
df_train.boxplot()
plt.title('Box plot of all columns')
plt.xticks(rotation=45)
plt.show()

# def remove_outliers(clean_df, threshold=3):
#     z_scores = np.abs((df_train - df_train.mean()) / df_train.std())
    
#     # Z 점수가 임계값보다 큰 데이터를 이상치로 간주하여 해당 열의 중앙값으로 대체
#     for col in df_train.columns:
#         mask = z_scores[col] > threshold
#         df_train.loc[mask, col] = df_train[col].median()
        
#     return clean_df

# # outlier 제거
# df_train = remove_outliers(df_train)


# ## 2-4 Correlation 파악

# In[1272]:


# 상관 행렬 계산
correlation_matrix = df_train.corr()
# 히트맵 그리기
plt.figure(figsize=(29, 29))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap of Rows')
plt.show()


# ## 2-2. 학습, 검증 데이터 분리

# In[1273]:


# 특성과 타겟 변수 선택
x = df_train.drop(columns=['is_converted'])  # 타겟 변수를 제외한 모든 열을 특성으로 선택
y = df_train['is_converted']  # 타겟 변수 선택

# 훈련 및 테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ## 3. 모델 학습

# ### 모델 정의 

# In[1274]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, random_state=0)


# ### 모델 학습

# In[1275]:


model.fit(x_train, y_train)


# ### 모델 성능 보기

# In[1276]:


y_pred = pd.DataFrame(model.predict(x_test))

def get_column_range(column):
    min_value = column.min()
    max_value = column.max()
    range_tuple = (min_value, max_value)
    return range_tuple

column_range = get_column_range(y_pred)
print("칼럼의 숫자들의 범위:", column_range)
print("칼럼의 숫자들의 평균:", y_pred.mean())
print("칼럼의 숫자들의 중앙값:", y_pred.median())

def convert_to_binary_based_on_median(column):
    """
    칼럼의 값이 0.5보다 큰 경우 1로, 작은 경우 0으로 변환
    """
    median_value = column.median()
    converted_column = (column > 0.5).astype(int)
    return converted_column

y_pred = convert_to_binary_based_on_median(y_pred)

def decode_binary(column):
    """
    컬럼의 값을 1은 True로, 0은 False로 디코딩합니다.
    """
    decoded_column = column == 1
    return decoded_column

# y_pred = decode_binary(y_pred)
# y_test = decode_binary(y_test)

# 성능 평가 함수 정의
def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))

# 성능 평가
get_clf_eval(y_test, y_pred)


# ## 4. 제출하기

# ### 테스트 데이터 예측

# In[1277]:


# 예측에 필요한 데이터 분리
x_test_data = df_test.drop(["is_converted", "id"], axis=1)


# In[1279]:


test_pred = model.predict(x_test_data.fillna(0))
def convert_to_binary_based_on_median(column):
    """
    칼럼의 값이 중앙값보다 큰 경우 1로, 작은 경우 0으로 변환
    """
    converted_column = (column > 0.5).astype(int)
    return converted_column

test_pred = convert_to_binary_based_on_median(test_pred)
print(test_pred)

def decode_binary(column):
    """
    컬럼의 값을 1은 True로, 0은 False로 디코딩
    """
    decoded_column = column == 1
    return decoded_column

test_pred = decode_binary(test_pred)
print(test_pred)
sum(test_pred) # True로 예측된 개수


# ### 제출 파일 작성

# In[1280]:


# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["is_converted"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)


# **우측 상단의 제출 버튼을 클릭해 결과를 확인하세요**

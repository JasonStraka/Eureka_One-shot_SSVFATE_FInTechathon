# encoding: utf-8

import os
import platform
import numpy as np
import pandas as pd
import torch
import copy
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models.ctr_model_utils import build_input_features, SparseFeat, DenseFeat, get_feature_names


class Avazu2party:

    def __init__(self, data_dir, data_type, k, input_size):
        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.k = k
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        dense_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                          'REGION_POPULATION_RELATIVE',
                          'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                          'EXT_SOURCE_1',
                          'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
                          'YEARS_BEGINEXPLUATATION_AVG',
                          'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
                          'FLOORSMIN_AVG',
                          'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                          'NONLIVINGAREA_AVG',
                          'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                          'COMMONAREA_MODE',
                          'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                          'NONLIVINGAREA_MODE',
                          'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                          'COMMONAREA_MEDI',
                          'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
                          'NONLIVINGAREA_MEDI',
                          'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE']

        sparse_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                           'CNT_CHILDREN',
                           'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                           'NAME_HOUSING_TYPE',
                           'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                           'FLAG_EMAIL',
                           'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                           'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                           'REG_REGION_NOT_WORK_REGION',
                           'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                           'LIVE_CITY_NOT_WORK_CITY',
                           'ORGANIZATION_TYPE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                           'ELEVATORS_MEDI',
                           'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                           'WALLSMATERIAL_MODE',
                           'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                           'OBS_60_CNT_SOCIAL_CIRCLE',
                           'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                           'FLAG_DOCUMENT_5',
                           'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                           'FLAG_DOCUMENT_10',
                           'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                           'FLAG_DOCUMENT_15',
                           'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                           'FLAG_DOCUMENT_20',
                           'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                           'AMT_REQ_CREDIT_BUREAU_WEEK',
                           'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

        sparse_features_list = [sparse_features[20:], sparse_features[:20]]
        dense_features_list = [dense_features[30:], dense_features[:30]]

        # train = pd.read_csv(os.path.join(self.data_dir, 'train_10W.txt'))  # , sep='\t', header=None)
        # train = pd.read_csv(os.path.join(self.data_dir, 'Train'))  # , sep='\t', header=None)
        # test = pd.read_csv(os.path.join(self.data_dir, 'Test'))  # , sep='\t', header=None)
        # # test = pd.read_csv(os.path.join(self.data_dir, 'test_2W.txt'))  # , sep='\t', header=None)
        # data = pd.concat([train, test], axis=0)
        data = pd.read_csv(os.path.join(self.data_dir, 'train_subset.csv'))  # , sep='\t', header=None)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = data[feat].astype(str)
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        # if platform.system() == 'Windows':
        #      train, test = train_test_split(data, test_size=0.2, shuffle=False)
        # else:
        train = data.iloc[:250000]
        test = data.iloc[250000:]

        if data_type.lower() == 'train':
            labels = train['TARGET']
        else:
            labels = test['TARGET']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                      for feat in dense_features_list[i]]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model
            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}
            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            self.x.append(x)
        del data, train, test

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx
        labels = []
        data = [self.x[0][indexx], self.x[1][indexx]]
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()


class CtrDataTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, percent=0.3, col_infos=None):
        self.percent = percent
        self.col_infos = col_infos
        self.masked_sparse_dim_list = [int(round(len(self.col_infos[i]['sparse_idx'])*self.percent)) for i in range(2)]
        self.masked_dense_dim_list = [int(round(len(self.col_infos[i]['dense_idx'])*self.percent)) for i in range(2)]

        self.sparse_idx_list = [self.col_infos[i]['sparse_idx'] for i in range(2)]
        self.dense_idx_list = [self.col_infos[i]['dense_idx'] for i in range(2)]
        self.sparse_voc_list = [np.array(self.col_infos[i]['sparse_voc']) for i in range(2)]

    def __call__(self, x, client_id=0):
        masked_sparse_dim = self.masked_sparse_dim_list[client_id]
        masked_dense_dim = self.masked_dense_dim_list[client_id]

        q = copy.deepcopy(x)

        if masked_sparse_dim > 0:
            masked_sparse_index = np.random.choice(self.sparse_idx_list[client_id], masked_sparse_dim,  replace=False)
            masked_sparse_value = [v - 1 for v in self.sparse_voc_list[client_id][masked_sparse_index]]
            q[masked_sparse_index] = masked_sparse_value

        if masked_dense_dim > 0:
            masked_dense_index = np.random.choice(self.dense_idx_list[client_id], masked_dense_dim,  replace=False)

            masked_dense_value = np.random.uniform(0.0, 1.0, len(masked_dense_index))
            q[masked_dense_index] = masked_dense_value

        return [x, q]


class AvazuAug2party:

    def __init__(self, data_dir, data_type, k, input_size):
        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.k = k
        train_ratio = 0.2
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        dense_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                          'REGION_POPULATION_RELATIVE',
                          'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                          'EXT_SOURCE_1',
                          'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
                          'YEARS_BEGINEXPLUATATION_AVG',
                          'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
                          'FLOORSMIN_AVG',
                          'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                          'NONLIVINGAREA_AVG',
                          'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                          'COMMONAREA_MODE',
                          'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                          'NONLIVINGAREA_MODE',
                          'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                          'COMMONAREA_MEDI',
                          'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
                          'NONLIVINGAREA_MEDI',
                          'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE']

        sparse_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                           'CNT_CHILDREN',
                           'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                           'NAME_HOUSING_TYPE',
                           'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                           'FLAG_EMAIL',
                           'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                           'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                           'REG_REGION_NOT_WORK_REGION',
                           'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                           'LIVE_CITY_NOT_WORK_CITY',
                           'ORGANIZATION_TYPE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                           'ELEVATORS_MEDI',
                           'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                           'WALLSMATERIAL_MODE',
                           'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                           'OBS_60_CNT_SOCIAL_CIRCLE',
                           'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                           'FLAG_DOCUMENT_5',
                           'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                           'FLAG_DOCUMENT_10',
                           'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                           'FLAG_DOCUMENT_15',
                           'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                           'FLAG_DOCUMENT_20',
                           'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                           'AMT_REQ_CREDIT_BUREAU_WEEK',
                           'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

        sparse_features_list = [sparse_features[20:], sparse_features[:20]]
        dense_features_list = [dense_features[30:], dense_features[:30]]
        # train = pd.read_csv(os.path.join(self.data_dir, 'Train'))  # , sep='\t', header=None)
        # # train = pd.read_csv(os.path.join(self.data_dir, 'train_10W.txt'))  # , sep='\t', header=None)
        # test = pd.read_csv(os.path.join(self.data_dir, 'Test'))  # , sep='\t', header=None)
        # # test = pd.read_csv(os.path.join(self.data_dir, 'test_2W.txt'))  # , sep='\t', header=None)
        # data = pd.concat([train, test], axis=0)
        data = pd.read_csv(os.path.join(self.data_dir, 'train_subset.csv'))  # , sep='\t', header=None)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = data[feat].astype(str)
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        train = data.iloc[:250000]
        test = data.iloc[250000:]

        if data_type.lower() == 'train':
            labels = train['TARGET']
        else:
            labels = test['TARGET']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        self.col_info_list = []

        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()+1, input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                      for feat in dense_features_list[i]]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model
            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}

            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            sparse_list = list(
                filter(lambda x: isinstance(x, SparseFeat), fixlen_feature_columns)) if len(
                fixlen_feature_columns) else []
            dense_list = list(
                filter(lambda x: isinstance(x, DenseFeat), fixlen_feature_columns)) if len(
                fixlen_feature_columns) else []
            sparse_idx = [feature_index[feat.name][0] for feat in sparse_list]
            dense_idx = [feature_index[feat.name][0] for feat in dense_list]
            sparse_voc_size = [feat.vocabulary_size for feat in sparse_list]
            self.col_info_list.append({'sparse_idx': sparse_idx, 'sparse_voc': sparse_voc_size, 'dense_idx': dense_idx})

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            self.x.append(x)
        self.transform = CtrDataTransform(0.3, self.col_info_list)

        del data, train, test

        print('*******************************',len(self.x[0]), len(self.x[1]))

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx
        labels = []
        data = []

        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = self.transform(x, i)
            data.append(x)

        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()



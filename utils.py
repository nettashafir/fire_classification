import os
import pickle
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

LABEL_COL = ['STAT_CAUSE_DESCR']
LEAK_COLS = ['STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR']

DROP_COLS = ['FOD_ID', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'NWCG_REPORTING_UNIT_ID',
             'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME',
             'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
             'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME', 'COMPLEX_NAME']


DROP_NO_USE = ['SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY', 'LATITUDE', 'LONGITUDE',
               'FIRE_SIZE_CLASS', 'OWNER_DESCR', 'STATE', 'COUNTY', 'Electric_Powerline_Dist', 'Railroads_Dist',
               'FIPS_CODE', 'FIPS_NAME', 'Shape', 'OBJECTID', 'date', 'geometry', 'month', 'year']


NON_BOOLEAN_FEATURES = ['FIRE_SIZE', 'geo_cluster_freq_Lightning', 'geo_cluster_freq_Powerline',
 'geo_cluster_freq_Smoking', 'geo_cluster_freq_Missing/Undefined', 'geo_cluster_freq_Debris Burning',
 'geo_cluster_freq_Arson', 'geo_cluster_freq_Equipment Use', 'geo_cluster_freq_Miscellaneous',
 'geo_cluster_freq_Railroad', 'geo_cluster_freq_Campfire', 'geo_cluster_freq_Children',
 'geo_cluster_freq_Structure', 'geo_cluster_freq_Fireworks', 'EASTINGS', 'NORTHINGS', 'DayofWeek',
 'sunrize', 'sunset', 'datetime_freq_Lightning', 'datetime_freq_Powerline', 'datetime_freq_Smoking',
 'datetime_freq_Missing/Undefined', 'datetime_freq_Debris Burning', 'datetime_freq_Arson',
 'datetime_freq_Equipment Use', 'datetime_freq_Miscellaneous', 'datetime_freq_Railroad', 'datetime_freq_Campfire',
 'datetime_freq_Children', 'datetime_freq_Structure', 'datetime_freq_Fireworks', 'FIRE_DURATION_HRS',
 'discovery_time_sin', 'discovery_time_cos', 'cont_time_sin', 'cont_time_cos', 'month_sin', 'month_cos',
 'temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max',
                        'et0_fao_evapotranspiration']


BOOLEAN_FEATURES = ['Cluster_number_0', 'Cluster_number_1', 'Cluster_number_2', 'Cluster_number_3', 'Cluster_number_4',
                    'Cluster_number_5', 'Cluster_number_6', 'Cluster_number_7', 'Cluster_number_8', 'Cluster_number_9',
                    'Cluster_number_10', 'Cluster_number_11', 'Cluster_number_12', 'Cluster_number_13',
                    'Cluster_number_14', 'Cluster_number_15', 'Cluster_number_16', 'Cluster_number_17',
                    'Cluster_number_18', 'Cluster_number_19', 'Cluster_number_20', 'Cluster_number_21',
                    'Cluster_number_22', 'Cluster_number_23', 'Cluster_number_24', 'Cluster_number_25',
                    'Cluster_number_26', 'Cluster_number_27', 'Cluster_number_28', 'Cluster_number_29',
                    'Cluster_number_30', 'Cluster_number_31', 'Cluster_number_32', 'Cluster_number_33',
                    'Cluster_number_34', 'Cluster_number_35', 'Cluster_number_36', 'Cluster_number_37',
                    'Cluster_number_38', 'Cluster_number_39', 'Cluster_number_40', 'Cluster_number_41',
                    'Cluster_number_42', 'Cluster_number_43', 'Cluster_number_44', 'Cluster_number_45',
                    'Cluster_number_46', 'Cluster_number_47', 'Cluster_number_48', 'Cluster_number_49',
                    'Cluster_number_50', 'Cluster_number_51', 'Cluster_number_52', 'Cluster_number_53',
                    'Cluster_number_54', 'Cluster_number_55', 'Cluster_number_56', 'Cluster_number_57',
                    'Cluster_number_58', 'Cluster_number_59', 'Cluster_number_60', 'Cluster_number_61',
                    'Cluster_number_62', 'Cluster_number_63', 'Cluster_number_64', 'Cluster_number_65',
                    'Cluster_number_66', 'Cluster_number_67', 'Cluster_number_68', 'Cluster_number_69',
                    'Cluster_number_70', 'Cluster_number_71', 'Cluster_number_72', 'Cluster_number_73',
                    'Cluster_number_74', 'Cluster_number_75', 'Cluster_number_76', 'Cluster_number_77',
                    'Cluster_number_78', 'Cluster_number_79','Cluster_number_80', 'Cluster_number_81',
                    'Cluster_number_82', 'Cluster_number_83', 'Cluster_number_84', 'Cluster_number_85',
                    'Cluster_number_86', 'Cluster_number_87', 'Cluster_number_88', 'Cluster_number_89',
                    'Cluster_number_90', 'Cluster_number_91', 'Cluster_number_92', 'Cluster_number_93',
                    'Cluster_number_94', 'Cluster_number_95', 'Cluster_number_96', 'Cluster_number_97',
                    'Cluster_number_98', 'Cluster_number_99', 'Cluster_number_100', 'Cluster_number_101',
                    'Cluster_number_102', 'Cluster_number_103', 'Cluster_number_104', 'Cluster_number_105',
                    'Cluster_number_106', 'Cluster_number_107', 'Cluster_number_108', 'Cluster_number_109',
                    'Cluster_number_110', 'Cluster_number_111', 'Cluster_number_112', 'Cluster_number_113',
                    'Cluster_number_114', 'Cluster_number_115', 'Cluster_number_116', 'Cluster_number_117',
                    'Cluster_number_118', 'Cluster_number_119', 'Cluster_number_120', 'Cluster_number_121',
                    'Cluster_number_122', 'Cluster_number_123', 'Cluster_number_124', 'Cluster_number_125',
                    'Cluster_number_126', 'Cluster_number_127', 'Cluster_number_128', 'Cluster_number_129',
                    'OWNER_CODE0', 'OWNER_CODE1', 'OWNER_CODE2', 'OWNER_CODE3', 'OWNER_CODE4', 'OWNER_CODE5',
                    'OWNER_CODE6', 'OWNER_CODE7', 'OWNER_CODE8', 'OWNER_CODE9', 'OWNER_CODE10', 'OWNER_CODE11',
                    'OWNER_CODE12', 'OWNER_CODE13', 'OWNER_CODE14', 'OWNER_CODE15', 'Electric_Powerline_Binary',
                    'Railroads_Binary', 'Indian_Lands_Inside', 'Military_Bases_Inside', 'National_Forests_Inside',
                    'National_Parks_Inside', 'Wildland_Urban_Inside', '1992', '1993', '1994', '1995', '1996', '1997',
                    '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006','2007', '2008', '2009',
                    '2010', '2011', '2012', '2013', '2014', '2015', 'IsWeekend', 'Christmas Day',
                    'Christmas Day (Observed)', 'Columbus Day', 'Independence Day', 'Independence Day (Observed)',
                    'Labor Day', 'Martin Luther King Jr. Day', 'Memorial Day', "New Year's Day",
                    "New Year's Day (Observed)", 'Thanksgiving', 'Veterans Day', 'Veterans Day (Observed)',
                    "Washington's Birthday", 'clear', 'rain', 'snow', 'fog', 'dust', 'thunder']

TEST_SIZE = 0.2
VAL_SIZE = 0.2

SQL_PATH = 'data/FPA_FOD_20170508.sqlite'
SQL_PARAMS = 'SELECT * from Fires'
WORK_DATA_PATH = 'data/split_data'

DATA_FILE_PREFIX = 'Data-'
LABELS_FILE_PREFIX = 'Labels-'

TRAIN_FILE_NAME = 'train.csv'
VAL_FILE_NAME = 'val.csv'
TEST_FILE_NAME = 'test.csv'

FITTED_CLASSES_DIR = './fitted_classes'


def get_work_data(work_data_path=WORK_DATA_PATH, sql_path=SQL_PATH,
                  val_size=VAL_SIZE, test_size=TEST_SIZE,
                  force_create=False):
    if not os.path.exists(work_data_path):
        print(f'creating work data dir at {work_data_path}...')
        os.makedirs(work_data_path)

    if os.listdir(work_data_path) and not force_create:
        return _load_work_data(work_data_path)

    return _create_work_data(work_data_path, sql_path, val_size, test_size)


def _create_work_data(work_data_path, sql_path, val_size, test_size):
    print(f'creating work data and saving at {work_data_path}...')
    # read the sql:
    conn = sqlite3.connect(sql_path)
    df = pd.read_sql_query(SQL_PARAMS, conn)

    # create the data:
    label_bin = LabelBinarizer()
    X, y = df.drop(columns=LABEL_COL + LEAK_COLS + DROP_COLS), label_bin.fit_transform(df[LABEL_COL])
    y = pd.DataFrame(y, columns=label_bin.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size))
    train, val, test = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # save the data:
    _save_work_data(train, val, test, work_data_path)

    return train, val, test


def _load_work_data(path):
    print(f'loading previously created work data from {path}...')
    return ((pd.read_csv(os.path.join(path, DATA_FILE_PREFIX + TRAIN_FILE_NAME), index_col=0),
             pd.read_csv(os.path.join(path, LABELS_FILE_PREFIX + TRAIN_FILE_NAME), index_col=0)),
            (pd.read_csv(os.path.join(path, DATA_FILE_PREFIX + VAL_FILE_NAME), index_col=0),
             pd.read_csv(os.path.join(path, LABELS_FILE_PREFIX + VAL_FILE_NAME), index_col=0)),
            (pd.read_csv(os.path.join(path, DATA_FILE_PREFIX + TEST_FILE_NAME), index_col=0),
             pd.read_csv(os.path.join(path, LABELS_FILE_PREFIX + TEST_FILE_NAME), index_col=0)))


def _save_work_data(train, val, test, path):
    train[0].to_csv(os.path.join(path, DATA_FILE_PREFIX + TRAIN_FILE_NAME))
    train[1].to_csv(os.path.join(path, LABELS_FILE_PREFIX + TRAIN_FILE_NAME))

    val[0].to_csv(os.path.join(path, DATA_FILE_PREFIX + VAL_FILE_NAME))
    val[1].to_csv(os.path.join(path, LABELS_FILE_PREFIX + VAL_FILE_NAME))

    test[0].to_csv(os.path.join(path, DATA_FILE_PREFIX + TEST_FILE_NAME))
    test[1].to_csv(os.path.join(path, LABELS_FILE_PREFIX + TEST_FILE_NAME))


def _get_features_creator(creator_class, X_train=None, y_train=None, force_fit=False):
    if not os.path.exists(FITTED_CLASSES_DIR):
        print(f'class instance dir at {FITTED_CLASSES_DIR}...')
        os.makedirs(FITTED_CLASSES_DIR)

    if (creator_class.FILENAME in os.listdir(FITTED_CLASSES_DIR)) and not force_fit:
        return creator_class.load()

    # assert (X_train is not None) and (y_train is not None) todo

    creator = creator_class()
    creator.fit(X_train, y_train)
    creator.save()

    return creator


def transform_features(creator_class, X_train, X_val, X_test,
                       y_train=None, y_val=None, y_test=None,
                       transform_y=False, force_fit=False):
    dt_features_creator = _get_features_creator(creator_class,
                                                X_train=X_train, y_train=y_train, force_fit=force_fit)
    if transform_y:
        print('Train')
        X_train, y_train = dt_features_creator.transform(X_train, y_train)
        print('Val')
        X_val, y_val = dt_features_creator.transform(X_val, y_val)
        print('Test')
        X_test, y_test = dt_features_creator.transform(X_test, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        print('Train')
        X_train = dt_features_creator.transform(X_train)
        print('Val')
        X_val = dt_features_creator.transform(X_val)
        print('Test')
        X_test = dt_features_creator.transform(X_test)
        return X_train, X_val, X_test


def load_final_data(sub_sample=1):
    X_train = pd.read_csv('./data/features_clean/X_train_wFeatures-clean.csv', index_col=0)[::sub_sample]
    y_train = pd.read_csv('./data/features_clean/y_train_wFeatures-clean.csv', index_col=0)[::sub_sample]

    X_val = pd.read_csv('./data/features_clean/X_val_wFeatures-clean.csv', index_col=0)[::sub_sample]
    y_val = pd.read_csv('./data/features_clean/y_val_wFeatures-clean.csv', index_col=0)[::sub_sample]

    X_test = pd.read_csv('./data/features_clean/X_test_wFeatures-clean.csv', index_col=0)[::sub_sample]
    y_test = pd.read_csv('./data/features_clean/y_test_wFeatures-clean.csv', index_col=0)[::sub_sample]
    return X_train, X_val, X_test, y_train, y_val, y_test


def pre_clean(X_train, X_val, X_test):
    X_train.drop(columns=DROP_NO_USE, inplace=True)
    X_val.drop(columns=DROP_NO_USE, inplace=True)
    X_test.drop(columns=DROP_NO_USE, inplace=True)

    print(f'before: train-{X_train.shape} val-{X_val.shape} test={X_test.shape}')
    X_train.dropna(inplace=True)
    X_val.dropna(inplace=True)
    X_test.dropna(inplace=True)
    print(f'after:  train-{X_train.shape} val-{X_val.shape} test={X_test.shape}')

    X_train = X_train.rename(columns={c: str(c) for c in X_train.columns})
    X_val = X_val.rename(columns={c: str(c) for c in X_val.columns})
    X_test = X_test.rename(columns={c: str(c) for c in X_test.columns})

    X_train[BOOLEAN_FEATURES] = X_train[BOOLEAN_FEATURES].astype(int)
    X_val[BOOLEAN_FEATURES] = X_val[BOOLEAN_FEATURES].astype(int)
    X_test[BOOLEAN_FEATURES] = X_test[BOOLEAN_FEATURES].astype(int)

    return X_train, X_val, X_test


def post_clean(X_train, X_val, X_test, X_train_non_bool, X_val_non_bool, X_test_non_bool):
    X_train = X_train.loc[X_train_non_bool.index]
    X_val = X_val.loc[X_val_non_bool.index]
    X_test = X_test.loc[X_test_non_bool.index]

    X_train[NON_BOOLEAN_FEATURES] = X_train_non_bool
    X_val[NON_BOOLEAN_FEATURES] = X_val_non_bool
    X_test[NON_BOOLEAN_FEATURES] = X_test_non_bool

    return X_train, X_val, X_test


def get_evaluation_data(train_path, test_path, val_size=0.2):
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    with open('./fitted_classes/label_bin.pkl', 'rb') as f:
        label_bin = pickle.load(f)

    X_train, y_train = train_df.drop(columns=LABEL_COL + LEAK_COLS + DROP_COLS), label_bin.transform(train_df[LABEL_COL])
    y_train = pd.DataFrame(y_train, columns=label_bin.classes_)

    X_test, y_test = test_df.drop(columns=LABEL_COL + LEAK_COLS + DROP_COLS), label_bin.transform(test_df[LABEL_COL])
    y_test = pd.DataFrame(y_test, columns=label_bin.classes_)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

    work_features = [c for c in X_train.columns if not c.startswith('Cluster_number')]

    X_train = X_train[work_features]
    X_val = X_val[work_features]
    X_test = X_test[work_features]

    y_train = pd.Series(y_train.values.argmax(1),index=X_train.index)
    y_val = pd.Series(y_val.values.argmax(1),index=X_val.index)
    y_test = pd.Series(y_test.values.argmax(1),index=X_test.index)

    return X_train, X_val, X_test, y_train, y_val, y_test


def predictions_to_labels(preds):
    with open('./fitted_classes/label_bin.pkl', 'rb') as f:
        label_bin = pickle.load(f)

    return label_bin.classes_[preds]

# if __name__ == '__main__':
    # get_work_data(force_create=True)
    # predictions_to_labels(None)
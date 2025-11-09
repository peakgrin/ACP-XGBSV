import argparse
import sys
import pandas as pd

path_train = './csvTrainSet/train_'
path_test  = './csvTestSet/test_'
feat_out = sys.argv[1]
feat_out_train = path_train + feat_out
feat_out_test  = path_test + feat_out
feat_in_1 = sys.argv[2]
feat_in_1_train = path_train + feat_in_1
feat_in_1_test  = path_test + feat_in_1
df_out_train =  pd.read_csv(feat_in_1_train, header=None)
df_out_test  =  pd.read_csv(feat_in_1_test, header=None)

for i in range(3,len(sys.argv)):
    feat_in_tmp_train = path_train + sys.argv[i]
    feat_in_tmp_test  = path_test + sys.argv[i]
    df_tmp_train = pd.read_csv(feat_in_tmp_train, header=None)
    df_tmp_train = df_tmp_train.drop(columns=df_tmp_train.columns[0])
    df_out_train = pd.concat([df_out_train, df_tmp_train], axis=1)
    df_tmp_test = pd.read_csv(feat_in_tmp_test, header=None)
    df_tmp_test = df_tmp_test.drop(columns=df_tmp_test.columns[0])
    df_out_test = pd.concat([df_out_test, df_tmp_test], axis=1)

df_out_train.to_csv(feat_out_train, index=False, header=None)
df_out_test.to_csv(feat_out_test, index=False, header=None)

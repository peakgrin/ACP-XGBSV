import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC as SVMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis

def feat_selection_selectKBest(num_feat, inFile, outFile, sign=1):
    path_train = './csvTrainSet/train_'
    path_test  = './csvTestSet/test_'
    inFile_train  = path_train + inFile
    inFile_test   = path_test  + inFile
    outFile_train = path_train + outFile
    outFile_test  = path_test  + outFile
    df = pd.read_csv(inFile_train, header=None)
    X_train = df.drop(columns=df.columns[0])
    y_train = df[df.columns[0]]
    if sign==1:
        selector = SelectKBest(k=num_feat)
        selector = selector.fit(X_train, y_train)
        scores = selector.scores_
        indices = np.argsort(scores)[::-1]
    elif sign==2:
        selector = SelectKBest(f_classif, k=num_feat)
        selector = selector.fit(X_train, y_train)
        scores = selector.scores_
        indices = np.argsort(scores)[::-1]
    elif sign==3:
        selector = SelectKBest(mutual_info_classif, k=num_feat)
        selector = selector.fit(X_train, y_train)
        scores = selector.scores_
        indices = np.argsort(scores)[::-1]
    elif sign==4:
        selector = SelectFromModel(DecisionTreeClassifier(random_state=1), threshold='median')
        selector.fit(X_train, y_train)
    elif sign==5:
        selector = RFE(estimator=DecisionTreeClassifier(random_state=1), n_features_to_select=num_feat, step=10, verbose=5)
        selector.fit(X_train, y_train)
    else:
        print('please choose a feature selection method')
        return
    selected_features = X_train.columns[selector.get_support()]
    X_train_new = selector.transform(X_train)
    X_train_new = pd.DataFrame(X_train_new)
    df_out_train = pd.concat([y_train, X_train_new], axis=1)
    df_out_train.to_csv(outFile_train, index=False, header=None)
    df = pd.read_csv(inFile_test, header=None)
    X_test = df.drop(columns=df.columns[0])
    y_test = df[df.columns[0]]
    X_test_new = selector.transform(X_test)
    X_test_new = pd.DataFrame(X_test_new)
    df_out_test = pd.concat([y_test, X_test_new], axis=1)
    df_out_test.to_csv(outFile_test, index=False, header=None)

def feat_selection_PCA(num_feat, inFile, outFile):
    inFile_train  = './csvTrainSet/train_' + inFile
    inFile_test   = './csvTestSet/test_' + inFile
    outFile_train = './csvTrainSet/train_' + outFile
    outFile_test  = './csvTestSet/test_' + outFile
    df = pd.read_csv(inFile_train, header=None)
    X_train = df.drop(columns=df.columns[0])
    y_train = df[df.columns[0]]
    pca = PCA(n_components=num_feat)
    pca.fit(X_train)
    X_pca = pca.transform(X_train)
    X_pca = pd.DataFrame(X_pca)
    df_out_train = pd.concat([y_train, X_pca], axis=1)
    df_out_train.to_csv(outFile_train, index=False, header=None)
    df = pd.read_csv(inFile_test, header=None)
    X_test = df.drop(columns=df.columns[0])
    y_test = df[df.columns[0]]
    X_pca = pca.transform(X_test)
    X_pca = pd.DataFrame(X_pca)
    df_out_test = pd.concat([y_test, X_pca], axis=1)
    df_out_test.to_csv(outFile_test, index=False, header=None)

if __name__ == '__main__':
    num_feat = sys.argv[1]
    inFile   = sys.argv[2]
    outFile  = sys.argv[3]
    sign     = sys.argv[4]
    if int(sign)==0:
        feat_selection_PCA(int(num_feat), inFile, outFile)
    else:
        feat_selection_selectKBest(int(num_feat), inFile, outFile, int(sign))

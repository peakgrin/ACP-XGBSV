import sys
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
import joblib
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def data_standard(X, sign_standard):
    if sign_standard==1:
        ss_X = StandardScaler()
        ss_X.fit(X)
        X = ss_X.transform(X)
    elif sign_standard==2:
        ss_X = MinMaxScaler()
        ss_X.fit(X)
        X = ss_X.transform(X)
    elif sign_standard==3:
        ss_X = Normalizer()
        ss_X.fit(X)
        X = ss_X.transform(X)
    elif sign_standard==4:
        ss_X = RobustScaler(quantile_range = (25,75))
        ss_X.fit(X)
        X = ss_X.transform(X)
    else:
        return None, None
    return ss_X, X

def train_test_ok(X_train, y_train, X_test, y_test, clf, sign_standard=1):
    if str(clf)=='XGBClassifier' or str(clf)=='XGBClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(eval_metric='mlogloss')
        else:
            clf = clf(eval_metric='mlogloss')
    elif str(clf)=='MLPClassifier' or str(clf)=='MLPClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(max_iter=2000)
        else:
            clf = clf(max_iter=2000)
    elif str(clf)=='LogisticRegression' or str(clf)=='LogisticRegression()':
        if isinstance(clf, str):
            clf = eval(clf)(max_iter=2000)
        else:
            clf = clf(max_iter=2000)
    elif str(clf)=='SVMClassifier' or str(clf)=='SVMClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(kernel='rbf',probability=True)
        else:
            clf = clf(kernel='rbf',probability=True)
    else:
        if isinstance(clf, str):
            clf = eval(clf)()
        else:
            clf = clf
    y = y_train
    X = X_train
    ss_X, X_train_tmp = data_standard(X, sign_standard)
    if ss_X is not None:
        X = X_train_tmp
    clf.fit(X,y)
    predicted_train = clf.predict(X)
    pred_prob_train = clf.predict_proba(X)
    y = y_test
    X = X_test
    if ss_X is not None:
        X = ss_X.transform(X)
    predicted_test = clf.predict(X)
    pred_prob_test = clf.predict_proba(X)
    return predicted_train, pred_prob_train, predicted_test, pred_prob_test

def crossValidate(X_train, y_train, clf, cv, rs, sign_standard=1, sign=2):
    y = y_train
    X = X_train
    if str(clf)=='XGBClassifier' or str(clf)=='XGBClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(eval_metric='mlogloss')
        else:
            clf = clf(eval_metric='mlogloss')
    elif str(clf)=='MLPClassifier' or str(clf)=='MLPClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(max_iter=2000)
        else:
            clf = clf(max_iter=2000)
    elif str(clf)=='LogisticRegression' or str(clf)=='LogisticRegression()':
        if isinstance(clf, str):
            clf = eval(clf)(max_iter=2000)
        else:
            clf = clf(max_iter=2000)
    elif str(clf)=='SVMClassifier' or str(clf)=='SVMClassifier()':
        if isinstance(clf, str):
            clf = eval(clf)(kernel='rbf',probability=True)
        else:
            clf = clf(kernel='rbf',probability=True)
    else:
        if isinstance(clf, str):
            clf = eval(clf)()
        else:
            clf = clf
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)
    ACC = []
    MCC = []
    SE = []
    SP = []
    precision = []
    f1_score = []
    AUC = []
    for train_index, test_index in kf.split(X,y):
        np.random.seed(rs)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        ss_X, X_train_tmp = data_standard(X_train, sign_standard)
        if ss_X is not None:
            X_train = X_train_tmp
            X_test = ss_X.transform(X_test)
        predicted_train, pred_prob_train, predicted_test, pred_prob_test = train_test_ok(X_train, y_train, X_test, y_test, clf, sign_standard)
        threshold = 0.5
        if sign == '2':
            clf_second=KNeighborsClassifier()
            clf_second.fit(pred_prob_train,y_train)
            pred = clf_second.predict(pred_prob_test)
            pred_prob = clf_second.predict_proba(pred_prob_test)
        elif sign == '3':
            clf_second=KNeighborsClassifier()
            clf_second.fit(pred_prob_train,y_train)
            pred_classifier = clf_second.predict(pred_prob_test)
            pred_prob_classifier = clf_second.predict_proba(pred_prob_test)
            pred_prob = (pred_prob_test + pred_prob_classifier)/2
            pred = (pred_prob[:,1]>=threshold+0).tolist()
        else:
            pred = predicted_test
            pred_prob = pred_prob_test
        TN, FP, FN, TP = metrics.confusion_matrix(y_test, pred).ravel()
        ACC.append( (TP+TN)/(TP+TN+FP+FN) )
        MCC.append( (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) )
        SE.append( TP/(TP+FN) )
        SP.append( TN/(TN+FP) )
        precision.append( TP/(TP+FP) )
        f1_score.append( 2*TP/(2*TP+FP+FN) )
        AUC.append( metrics.roc_auc_score(y_test, pred_prob[:,1]) )
    ACC_mean = np.mean(ACC)
    MCC_mean = np.mean(MCC)
    SE_mean = np.mean(SE)
    SP_mean = np.mean(SP)
    precision_mean = np.mean(precision)
    f1_score_mean = np.mean(f1_score)
    AUC_mean = np.mean(AUC)
    print('ACC: ' + str(ACC_mean) )
    print('MCC: ' + str(MCC_mean) )
    print('SE: ' + str(SE_mean) )
    print('SP: ' + str(SP_mean) )
    print('precision: ' + str(precision_mean) )
    print('f1_score: ' + str(f1_score_mean) )
    print('AUC: ' + str(AUC_mean) )
    print()
    return ACC_mean, MCC_mean, SE_mean, SP_mean, precision_mean, f1_score_mean, AUC_mean

def vote(rs=1, sign=1, *args):
    data_label_test = args[0][3]
    data_label_train = args[0][1]
    results = []
    for item in args:
        X_train = item[0]
        y_train = item[1]
        X_test = item[2]
        y_test = item[3]
        np.random.seed(rs)
        predicted_train, pred_prob_train, predicted_test, pred_prob_test = train_test_ok(X_train, y_train, X_test, y_test, item[4])
        results.append([pred_prob_train, pred_prob_test, predicted_train, predicted_test])
    pred_test_vote = []
    pred_prob_test_vote = []
    results_pred = []
    results_pred_prob = []
    for item in results:
        results_pred.append(item[3])
        results_pred_prob.append(item[1])
    results_pred = np.array(results_pred)
    results_pred_prob = np.array(results_pred_prob,dtype='float32')
    threshold = 0.5
    if sign == '0':
        print('sign==0, vote: hard')
        pred_prob_test_vote = results_pred_prob.mean(axis=0)
        pred_test_vote = (((results_pred.mean(axis=0))>=threshold)+0).tolist()
    elif sign == '1':
        print('sign==1, vote: soft')
        print('threshold = ' + str(threshold))
        pred_prob_test_vote = results_pred_prob.mean(axis=0)
        pred_test_vote = (pred_prob_test_vote[:,1]>=threshold+0).tolist()
    else:
        pass
    pred_prob_train = []
    pred_prob_test = []
    pred_train = []
    pred_test = []
    for item in results:
        pred_prob_train.append(item[0][:,1])
        pred_prob_test.append(item[1][:,1])
    pred_prob_train = np.array(pred_prob_train).T
    pred_prob_test = np.array(pred_prob_test).T
    data_label_train = np.array(data_label_train)
    data_label_test = np.array(data_label_test)
    clf_second=KNeighborsClassifier()
    clf_second.fit(pred_prob_train,data_label_train)
    pred_test_classifier = clf_second.predict(pred_prob_test)
    pred_prob_test_classifier = clf_second.predict_proba(pred_prob_test)
    if sign == '0' or sign == '1':
        pred_prob = pred_prob_test_vote
        pred = pred_test_vote
    elif sign == '2':
        pred_prob = pred_prob_test_classifier
        pred = pred_test_classifier
    elif sign == '3':
        pred_prob = results_pred_prob.mean(axis=0)
        pred_prob = (pred_prob + pred_prob_test_classifier)/2
        pred = (pred_prob[:,1]>=threshold+0).tolist()
    else:
        pass
    TN, FP, FN, TP = metrics.confusion_matrix(data_label_test, pred).ravel()
    print('(TN,FP,FN,TP)={},{},{},{}'.format(TN,FP,FN,TP) )
    print('TN+TP: '+ str(TN+TP) )
    ACC = (TP+TN)/(TP+TN+FP+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)
    precision = TP/(TP+FP)
    f1_score = 2*TP/(2*TP+FP+FN)
    AUC = metrics.roc_auc_score(data_label_test, pred_prob[:,1])
    return ACC,MCC,SE,SP,precision,f1_score,AUC

def vote_cv(rs=1, sign=1, *args):
    data_label_test = args[0][3]
    data_label_train = args[0][1]
    results = []
    for item in args:
        X_train = item[0]
        y_train = item[1]
        X_test = item[2]
        y_test = item[3]
        np.random.seed(rs)
        predicted_train, pred_prob_train, predicted_test, pred_prob_test = train_test_ok(X_train, y_train, X_test, y_test, item[4])
        results.append([pred_prob_train, pred_prob_test, predicted_train, predicted_test])
    pred_test_vote = []
    pred_prob_test_vote = []
    results_pred = []
    results_pred_prob = []
    for item in results:
        results_pred.append(item[3])
        results_pred_prob.append(item[1])
    results_pred = np.array(results_pred)
    results_pred_prob = np.array(results_pred_prob,dtype='float32')
    threshold = 0.5
    if sign == '0':
        pred_prob_test_vote = results_pred_prob.mean(axis=0)
        pred_test_vote = (((results_pred.mean(axis=0))>=threshold)+0).tolist()
    elif sign == '1':
        pred_prob_test_vote = results_pred_prob.mean(axis=0)
        pred_test_vote = (pred_prob_test_vote[:,1]>=threshold+0).tolist()
    else:
        pass
    pred_prob_train = []
    pred_prob_test = []
    pred_train = []
    pred_test = []
    for item in results:
        pred_prob_train.append(item[0][:,1])
        pred_prob_test.append(item[1][:,1])
    pred_prob_train = np.array(pred_prob_train).T
    pred_prob_test = np.array(pred_prob_test).T
    data_label_train = np.array(data_label_train)
    data_label_test = np.array(data_label_test)
    clf_second=KNeighborsClassifier()
    clf_second.fit(pred_prob_train,data_label_train)
    pred_test_classifier = clf_second.predict(pred_prob_test)
    pred_prob_test_classifier = clf_second.predict_proba(pred_prob_test)
    if sign == '0' or sign == '1':
        pred_prob = pred_prob_test_vote
        pred = pred_test_vote
    elif sign == '2':
        pred_prob = pred_prob_test_classifier
        pred = pred_test_classifier
    elif sign == '3':
        pred_prob = results_pred_prob.mean(axis=0)
        pred_prob = (pred_prob + pred_prob_test_classifier)/2
        pred = (pred_prob[:,1]>=threshold+0).tolist()
    else:
        pass
    TN, FP, FN, TP = metrics.confusion_matrix(data_label_test, pred).ravel()
    ACC = (TP+TN)/(TP+TN+FP+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)
    precision = TP/(TP+FP)
    f1_score = 2*TP/(2*TP+FP+FN)
    AUC = metrics.roc_auc_score(data_label_test, pred_prob[:,1])
    return ACC,MCC,SE,SP,precision,f1_score,AUC

def main(rs=1, sign_standard=1):
    print("==========================================")
    sign = sys.argv[2]
    clf = sys.argv[3]
    model_name = sys.argv[3]
    print('Classifier is: ' + clf)
    print('model filename is: ' + model_name)
    print()
    for i in range(4,len(sys.argv)):
        np.random.seed(rs)
        print("train datafile is: ./csvTrainSet/train_" + sys.argv[i])
        print("test datafile is: ./csvTestSet/test_" + sys.argv[i])
        data_train_path = './csvTrainSet/train_' + sys.argv[i]
        data_test_path = './csvTestSet/test_' + sys.argv[i]
        df_train = pd.read_csv(data_train_path, header=None)
        df_train_values = df_train.values[:,:]
        y_train = df_train_values[:, 0]
        X_train = df_train_values[:, 1:]
        df_test = pd.read_csv(data_test_path, header=None)
        df_test_values = df_test.values[:,:]
        y_test = df_test_values[:, 0]
        X_test = df_test_values[:, 1:]
        predicted_train, pred_prob_train, predicted_test, pred_prob_test = train_test_ok(X_train, y_train, X_test, y_test, clf, sign_standard)
        threshold = 0.5
        if sign == '2':
            clf_second=KNeighborsClassifier()
            clf_second.fit(pred_prob_train,y_train)
            pred = clf_second.predict(pred_prob_test)
            pred_prob = clf_second.predict_proba(pred_prob_test)
        elif sign == '3':
            clf_second=KNeighborsClassifier()
            clf_second.fit(pred_prob_train,y_train)
            pred_classifier = clf_second.predict(pred_prob_test)
            pred_prob_classifier = clf_second.predict_proba(pred_prob_test)
            pred_prob = (pred_prob_test + pred_prob_classifier)/2
            pred = (pred_prob[:,1]>=threshold+0).tolist()
        else:
            pred = predicted_test
            pred_prob = pred_prob_test
        TN, FP, FN, TP = metrics.confusion_matrix(y_test, pred).ravel()
        print('(TN,FP,FN,TP)={},{},{},{}'.format(TN,FP,FN,TP) )
        print('TN+TP: '+ str(TN+TP) )
        ACC = (TP+TN)/(TP+TN+FP+FN)
        MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        SE = TP/(TP+FN)
        SP = TN/(TN+FP)
        precision = TP/(TP+FP)
        f1_score = 2*TP/(2*TP+FP+FN)
        AUC = metrics.roc_auc_score(y_test, pred_prob[:,1])
        print('ACC: ' + str(ACC) )
        print('MCC: ' + str(MCC) )
        print('SE: ' + str(SE) )
        print('SP: ' + str(SP) )
        print('precision: ' + str(precision) )
        print('f1_score: ' + str(f1_score) )
        print('AUC: ' + str(AUC) )
        print()

def main_cv(cv=10, rs=1, sign_standard=1):
    print("==========================================")
    sign = sys.argv[2]
    clf = sys.argv[3]
    model_name = sys.argv[3]
    print('Classifier is: ' + clf)
    for i in range(4,len(sys.argv)):
        np.random.seed(rs)
        print("train datafile is: ./csvTrainSet/train_" + sys.argv[i])
        data_train_path = './csvTrainSet/train_' + sys.argv[i]
        df_train = pd.read_csv(data_train_path, header=None)
        df_train_values = df_train.values[:,:]
        y_train = df_train_values[:, 0]
        X_train = df_train_values[:, 1:]
        crossValidate(X_train, y_train, clf, cv, rs, sign_standard, sign)

def main_vote(rs=1, sign_standard=1):
    print("==========================================")
    sign = sys.argv[2]
    arg_4vote = []
    arg_4vote.append(sign)
    for i in range(3,len(sys.argv),2):
        np.random.seed(rs)
        clf = sys.argv[i]
        print('Classifier {} is: {} '.format(int((i-1)/2), clf) )
        data_train_path = './csvTrainSet/train_' + sys.argv[i+1]
        data_test_path = './csvTestSet/test_' + sys.argv[i+1]
        print('datafile {} is  : {}'.format(int((i-1)/2), sys.argv[i+1]) )
        df_train = pd.read_csv(data_train_path, header=None)
        df_train_values = df_train.values[:,:]
        y_train = df_train_values[:, 0]
        X_train = df_train_values[:, 1:]
        df_test = pd.read_csv(data_test_path, header=None)
        df_test_values = df_test.values[:,:]
        y_test = df_test_values[:, 0]
        X_test = df_test_values[:, 1:]
        arg_4vote.append([X_train, y_train, X_test, y_test, clf])
    ACC,MCC,SE,SP,precision,f1_score,AUC = vote(rs,*arg_4vote)
    print('ACC: ' + str(ACC) )
    print('MCC: ' + str(MCC) )
    print('SE: ' + str(SE) )
    print('SP: ' + str(SP) )
    print('precision: ' + str(precision) )
    print('f1_score: ' + str(f1_score) )
    print('AUC: ' + str(AUC) )

def main_vote_cv(cv=10, rs=1, sign_standard=1):
    print("==========================================")
    sign = sys.argv[2]
    data_train_path = './csvTrainSet/train_' + sys.argv[4]
    df_train = pd.read_csv(data_train_path, header=None)
    df_train_values = df_train.values[:,:]
    y = df_train_values[:, 0]
    X = df_train_values[:, 1:]
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)
    ACC = []
    MCC = []
    SE = []
    SP = []
    precision = []
    f1_score = []
    AUC = []
    for i in range(3,len(sys.argv),2):
        np.random.seed(rs)
        clf = sys.argv[i]
        print('Classifier {} is: {} '.format(int((i-1)/2), clf) )
        data_train = './csvTrainSet/train_' + sys.argv[i+1]
        print("datafile {} is  : {}".format(int((i-1)/2), sys.argv[i+1]) )
    for train_index, test_index in kf.split(X,y):
        np.random.seed(rs)
        arg_4vote = []
        arg_4vote.append(rs)
        arg_4vote.append(sign)
        for i in range(3,len(sys.argv),2):
            clf = sys.argv[i]
            data_train_path = './csvTrainSet/train_' + sys.argv[i+1]
            df_train = pd.read_csv(data_train_path, header=None)
            df_train_values = df_train.values[:,:]
            y_tmp = df_train_values[:, 0]
            X_tmp = df_train_values[:, 1:]
            X_train, y_train = X_tmp[train_index], y_tmp[train_index]
            X_test, y_test = X_tmp[test_index], y_tmp[test_index]
            arg_4vote.append([X_train, y_train, X_test, y_test, clf])
        ACC_tmp,MCC_tmp,SE_tmp,SP_tmp,precision_tmp,f1_score_tmp,AUC_tmp = vote_cv(*arg_4vote)
        ACC.append(ACC_tmp)
        MCC.append(MCC_tmp)
        SE.append(SE_tmp)
        SP.append(SP_tmp)
        precision.append(precision_tmp)
        f1_score.append(f1_score_tmp)
        AUC.append(AUC_tmp)
    ACC_mean = np.mean(ACC)
    MCC_mean = np.mean(MCC)
    SE_mean = np.mean(SE)
    SP_mean = np.mean(SP)
    precision_mean = np.mean(precision)
    f1_score_mean = np.mean(f1_score)
    AUC_mean = np.mean(AUC)
    print('ACC: ' + str(ACC_mean) )
    print('MCC: ' + str(MCC_mean) )
    print('SE: ' + str(SE_mean) )
    print('SP: ' + str(SP_mean) )
    print('precision: ' + str(precision_mean) )
    print('f1_score: ' + str(f1_score_mean) )
    print('AUC: ' + str(AUC_mean) )

sign_as = int(sys.argv[1])
if sign_as==1:
    main()
elif sign_as==2:
    main_cv()
elif sign_as==3:
    main_vote()
elif sign_as==4:
    main_vote_cv()
else:
    print('Please choose a main function.')

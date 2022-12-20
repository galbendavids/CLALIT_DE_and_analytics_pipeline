import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pickle
import sys
from imblearn.under_sampling import RandomUnderSampler
import shap
def my_scoring_method(y_true,y_pred, another_argument):
    #from the reinforcnemtn learning world - giving model aware

    mcm = confusion_matrix(y_true, y_pred)
    tn = mcm[0, 0]
    fn = mcm[1, 0]
    fp = mcm[0, 1]
    tp = mcm[1, 1]

    BHR=(tp+tn-fn-fp)/(tp+tn+fp+fn)

    return BHR

index = 2  #2 or 3
metrics="f1_score"
selected_scoring_method=f1_score
with_sampling= True
problem='PRESUMED_NAFLD'
more_info="_1_steps"
print("****")
print("****")
desc_string="xgboost - 70-30, "+metrics+", index="+str(index)+", "+problem+"_with-sampling:"+str(with_sampling)+more_info
print(desc_string)




string = "index=" + str(index) + "___" + problem
print(string)
if index==2:
    df_ = pd.read_csv("outputs/dataset_2_.csv", low_memory=False)
else:
    df_ = pd.read_csv("outputs/dataset_3_.csv", low_memory=False)
df = df_.drop(columns=['Unnamed: 0', 'PID'])
data_type = df.dtypes

for col in df.columns:
    if data_type[col] == 'O':
        print(col, flush=True)
        print(data_type[col], flush=True)
        # df[col]=df[col].astype('int')
        df[col] = pd.to_numeric(df[col], errors='coerce')

if problem == "PRESUMED_NAFLD":
    # PRESUMED NAFLD
    PRE_NAFLD_DF = df[[c for c in df if c not in ['PRESUMED_NAFLD', 'NAFLD']] + ['PRESUMED_NAFLD']]
    used_df = PRE_NAFLD_DF
else:
    # NAFLD
    NAFLD_DF = df[[c for c in df if c not in ['PRESUMED_NAFLD', 'NAFLD']] + ['NAFLD']]
    used_df = NAFLD_DF

X = used_df.iloc[:, :-1]
y = used_df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, stratify=y)


dataset_combined=X_test.copy()
dataset_combined[problem]=Y_test


from collections import Counter

print(sorted(Counter(Y_train).items()))

print(sorted(Counter(Y_test).items()))

from imblearn.under_sampling import RandomUnderSampler



if with_sampling:
    if index==2:
        rus = RandomUnderSampler( {1: 5432, 0: 5432*4}) #for index=2, Presumed
        X_resampled, y_resampled = rus.fit_resample(X_train,Y_train)
    if index==3:
        rus = RandomUnderSampler( {1: 5996, 0: 5996*4}) #for index=2, Presumed
        X_resampled, y_resampled = rus.fit_resample(X_train,Y_train)
    else:
        "something is wrong"
else:
    #rus = RandomUnderSampler( {1: 5432, 0: 37032}) #w/o sampling
    X_resampled, y_resampled = X_train, Y_train



#index 3 - presumed
#{'scale_pos_weight': 100, 'n_estimators': 1000, 'min_child_weight': 8, 'max_depth': 3, 'lambda': 3, 'gamma': 0.4, 'eta': 0.5, 'colsample_bytree': 0.7}
#???
#xgbc = XGBClassifier(gamma=0, learning_rate=0.3, max_depth=27, n_estimators=400, scale_pos_weight=100000)
xgbc = XGBClassifier(scale_pos_weight=20, n_estimators=1500, max_depth=5, min_child_weight=8, reg_lambda=2, gamma=0.3, eta=0.4, colsample_bytree=0.85)

xgbc.fit(X_resampled, y_resampled)
print("for index:" + str(index), flush=True)

ypred = xgbc.predict(X_test)
cm = confusion_matrix(Y_test, ypred)
print(cm, flush=True)
print(" ")
print("my_scoring_method:")
print(my_scoring_method(Y_test, ypred,0))
print(" ")
filename = "outputs/simple_model" + string + "_.sav"
pickle.dump(xgbc, open(filename, 'wb'))



clf = xgbc

#parameters = {
#     "lambda": [ 1, 2, 3],
#     "eta": [  0.25, 0.30, 0.35, 0.4, 0.5, 0.6],
#     "max_depth": [3, 4, 5, 6, 8],
#     "min_child_weight": [4, 5, 7, 8],
#     "gamma": [ 0.3, 0.4, 0.5],
#     "colsample_bytree": [  0.7, 0.8, 0.85, 0.9],
#     "n_estimators": [ 1000,1200,1500,2000,2500],
#     "scale_pos_weight": [5,10,20, 100, 1000, 10000, 50000, 100000, 150000]
# }

parameters = {
    "lambda": [ 1],
    "eta": [0.30],
    "max_depth": [4],
    "min_child_weight": [5],
    "gamma": [ 0.3],
    "colsample_bytree": [ 0.8],
    "n_estimators": [2500],
    "scale_pos_weight": [5]
}



clf = RandomizedSearchCV(estimator=xgbc,
                         param_distributions=parameters,
                         scoring=selected_scoring_method,
                         n_iter=22,
                         verbose=1,
                         cv=3,
                         n_jobs=-1)
#n_iter=22

clf.fit(X_resampled, y_resampled)
print("Best parameters:", flush=True)
print(clf.best_params_, flush=True)
print("Lowest roc_auc_score: ",flush=True)
print(clf.best_score_, flush=True)

print(clf.cv_results_,flush=True)
print(clf.best_params_,flush=True)
best_model = clf.best_estimator_
filename = "outputs/best_model_aug" + string + "_cv=3__.sav"
pickle.dump(best_model, open(filename, 'wb'))

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc', 'balanced_accuracy',
           'precision_weighted', 'recall_weighted']

#scores = cross_validate(best_model, X, y, scoring=scoring, cv=5, return_estimator=True)
scores = cross_validate(best_model, X_resampled, y_resampled, scoring=scoring, cv=5, return_estimator=True,n_jobs=-1)
print("best model result:",flush=True)
sorted(scores.keys())
fit_time = scores['fit_time']
print("fit time",flush=True)
print(fit_time,flush=True)
accuracy = scores['test_accuracy']
print("Accuracy",flush=True)
print(accuracy,flush=True)
precision = scores['test_precision_macro']
print("Presicion",flush=True)
print(precision,flush=True)
recall = scores['test_recall_macro']
print("recall",flush=True)
print(recall,flush=True)
print("f1",flush=True)
f1 = scores['test_f1_weighted']
print(f1,flush=True)

roc = scores['test_roc_auc']
print("roc",flush=True)
print(roc,flush=True)
ba = scores['test_balanced_accuracy']
print("ba",flush=True)
print(ba,flush=True)
#scores.to_csv("outputs/final_models_cv=5_" + desc_string + "_.csv")
print(scores,flush=True)
estimators=scores['estimator']
precision=scores['test_precision_macro']
max_index = np.argmax(precision, axis=0)
prediction_model=estimators[max_index]



filename = "outputs/final_model_"+desc_string+"_cv=5_(n)"+ "_.sav"
pickle.dump(prediction_model, open(filename, 'wb'))
print("finish -"+desc_string,flush=True)

#prediction model is the original one - cv=5
#another one with cv=3
#p means paralel programing - doesnt need to affect results
#(newer_sh) - is for shorter random search - only 22 steps

print("cm_for_final_model"+desc_string, flush=True)

ypred = prediction_model.predict(X_test)
cm = confusion_matrix(Y_test, ypred)
print(cm, flush=True)
filename = "outputs/final_model_"+ desc_string+"_.sav"




# explainer = shap.Explainer(prediction_model)
# shap_values = explainer(X_resampled)
# shap.plots.waterfall(shap_values[0])

print("finish !! yes! ")

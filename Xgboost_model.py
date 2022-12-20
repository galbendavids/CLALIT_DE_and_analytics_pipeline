import pickle
import math
import pandas as pd
import numpy as np
import datetime

cut_year=2007
date = datetime.date(cut_year, 1, 1)

path="/gpfs0/bgu-nadavrap/users/galbend/"
#old_df=pd.read_csv(path+"new_model_table2402.csv")

m_merged=pd.read_csv(path+"merged_dataset_"+str(date)+".csv")
#m_merged=pd.read_csv("/Users/galbd/Documents/thesis/clalit/")

def show_table(df):
    cols=df.columns
    for i in cols:
        print  (i)
    print(df.info())
    print(df.dtypes)

show_table(m_merged)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate

y=new_model_table[['DiagnosisCode']]
X=new_model_table.drop(columns=['DiagnosisCode'])
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)
#30 was ok..
xgbc = XGBClassifier(gamma=0, learning_rate=0.1,max_depth=25,n_estimators=100,scale_pos_weight=30)
print(xgbc)
xgbc.fit(X_train, Y_train)
ypred = xgbc.predict(X_test)
cm = confusion_matrix(Y_test,ypred)
print(cm)


scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc','balanced_accuracy','precision_weighted','recall_weighted']
scores = cross_validate(xgbc, X_train, Y_train, scoring=scoring, cv=4,return_estimator=True)
sorted(scores.keys())
fit_time = scores['fit_time']
score_time = scores['score_time']
accuracy = scores['test_accuracy']
precision = scores['test_precision_macro']
recall = scores['test_recall_macro']
f1 = scores['test_f1_weighted']
roc = scores['test_roc_auc']
ba = scores['test_balanced_accuracy']

print(accuracy)
print(precision)
print(recall)
print(roc)
print(ba)


#####

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt

def confusion_matrix(prediction_model):
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(prediction_model, X_test, Y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

def best_model(scores):
    estimators=scores['estimator']
    recall=scores['test_precision_macro']
    max_index = np.argmax(recall, axis=0)
    prediction_model=estimators[max_index]
    return prediction_model

prediction_model = best_model(scores)
confusion_matrix(prediction_model)

plot_confusion_matrix(prediction_model, X_test, Y_test,cmap=plt.cm.Blues)
plt.show()

from xgboost import plot_importance
from matplotlib import pyplot

pyplot.figure(figsize=(5000,1440))
plot_importance(prediction_model,max_num_features=16)
pyplot.show()

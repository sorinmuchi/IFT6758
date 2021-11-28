#!/usr/bin/env python
# coding: utf-8


from comet_ml import Experiment
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv('../.env')
COMET_API_KEY = os.getenv('COMET_API_KEY')
exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from numpy import sort
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibrationDisplay
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import seaborn as sn
from matplotlib.pyplot import figure
from sklearn import linear_model
from sklearn import feature_selection 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from xgboost import plot_importance
import xgboost as xgb
from matplotlib import pyplot
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import numpy

import shap

from joblib import dump
from comet_ml import API
# Read and preprocess data
df = pd.read_csv('../data/trainingSet.csv')
display(df.head())
df["rebound"]=df['rebound'].fillna(value=0)

df['rebound']=df['rebound'].astype(int)

FinalDf = df[['gameSeconds','period','xCoord','yCoord','distanceFromNet','angle','shotType','lastEventType', 'lastEventXCoord',
       'lastEventYCoord','lastEventGameSeconds','distanceFromLastEvent','rebound', 
       'changeInAngleShot', 'speed','Goal']]
obj_df=FinalDf.select_dtypes(include=['object']).copy()


FinalDf=FinalDf.drop(obj_df.columns, axis=1)

obj_df=obj_df.apply(preprocessing.LabelEncoder().fit_transform)

#Frame is the dataset used fpr xgboost later
Frame=pd.concat([FinalDf,obj_df],axis=1)

 
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(FinalDf)
Transformed_Values=imp_mean.transform(FinalDf)
TransformedDf = pd.DataFrame(Transformed_Values, index=df.index, columns=FinalDf.columns)


# ## Analyzing correlation between:  
# **features to features (redundancy identification) 
# **features to target (relevance identification)


plt.figure(figsize=(18,18))

correlation_mat = FinalDf.corr()
#sn.cubehelix_palette(as_cmap=True, reverse=True)
svm=sn.heatmap(correlation_mat, annot = True,cmap='rocket_r')

plt.savefig('correlation.png', dpi=400)


# ####  Split the data , this split is performed only once (same events will be used as train/test for each model) 




TransformedDf_=TransformedDf.drop("period",axis=1)
X = TransformedDf_.drop('Goal',axis=1)
y = TransformedDf_['Goal'].to_numpy()
X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)



def compute_goal_rate_per_percentile(probs, y):
    percentiles = []
    rates = []

    for i in range(0, 101):
        percentile = np.percentile(probs, i)
        r_percentile = np.float32(round(percentile, 2))
        goals = 0
        no_goals = 0
        for idx, p in enumerate(probs):
#             if p<=percentile:
            if np.float32(round(p, 2))==r_percentile:
                if y[idx]==1:
                    goals+=1
                else:
                    no_goals+=1
        if (goals+no_goals==0):
            rate=0
        else:
            rate = goals *100 / (goals + no_goals)
        percentiles.append(percentile)
        rates.append(rate)
    return percentiles, rates


def compute_cumulative_propotion_of_goals_per_percentile(probs, y):
    percentiles = []
    cum_rates = []
    cum_rate = 0
    total_goals = sum(y)
    cum_goals = 0
    for i in range(0, 101):
        percentile = np.percentile(probs, i)
        cum_goals=0
        for idx, p in enumerate(probs):
            if p<=percentile:
                if y[idx]==1:
                    cum_goals+=1 
        cum_rate = cum_goals * 100 / total_goals
        percentiles.append(percentile)
        cum_rates.append(cum_rate)
    return percentiles, cum_rates



def displayFigures(Models,X_train,Y_train,X_val,Y_val,Text_label,X_trainAll=X_train,Y_trainAll=y_train,X_valAll=x_val,Y_valAll=y_val):
    
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    #RandomClassifier
    random_clf = DummyClassifier(strategy="uniform").fit(X_trainAll, Y_trainAll)

    train_score2 = random_clf.score(X_trainAll, Y_trainAll)
    val_score2 = random_clf.score(X_valAll, Y_valAll)

    print(f'Training accuracy: {train_score2}')
    print(f'Validation accuracy: {val_score2}')
    
    random_probs = random_clf.predict_proba(X_valAll.to_numpy().reshape(-1, len(X_valAll.columns))[:, :])[:,1]

    random_auc = roc_auc_score(Y_valAll, random_probs)

    print('Random: ROC AUC=%.3f' % (random_auc))


    random_fpr, random_tpr, _ = roc_curve(Y_valAll, random_probs)
    axs[0, 0].plot(random_fpr, random_tpr, linestyle='--', marker='.', label='Random')
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    
    
    percentiles2, rates2 = compute_goal_rate_per_percentile(random_probs, Y_valAll)
    axs[0, 1].plot(percentiles2, rates2, marker='.', label='Random')
    axs[0, 1].set_xlabel('Shot probability model percentile')
    axs[0, 1].set_ylabel('Goal rate')
    
    
    percentiles2, rates2 = compute_cumulative_propotion_of_goals_per_percentile(random_probs, Y_valAll)
    axs[1, 0].plot(percentiles2, rates2, marker='.', label='Random')
    axs[1, 0].set_xlabel('Shot probability model percentile')
    axs[1, 0].set_ylabel('Cumulative proportion of goals')
     
    
    disp4 = CalibrationDisplay.from_estimator(random_clf, X_valAll, Y_valAll, label='Random', ax=axs[1, 1])

    for idx, Model in enumerate(Models):
        print(Text_label[idx])
        xgb_probs = Model.predict_proba(X_val[idx][:, :])[:,1]
        xgb_auc = roc_auc_score(Y_val[0], xgb_probs)
        xgb_acc=Model.score(X_val[idx],Y_val[0])
        print(Text_label[idx]+': ROC AUC=%.3f' % (xgb_auc))
        print(Text_label[idx]+': Accuracy=%.3f' % (xgb_acc))

        xgb_fpr1, xgb_tpr1, _ = roc_curve(Y_val[0], xgb_probs)


        axs[0, 0].plot(xgb_fpr1, xgb_tpr1, marker='.', label=Text_label[idx])


        percentiles1, rates1 = compute_goal_rate_per_percentile(xgb_probs, Y_val[0])


        axs[0, 1].plot(percentiles1, rates1, marker='.', label=Text_label[idx])

        percentiles1, rates1 = compute_cumulative_propotion_of_goals_per_percentile(xgb_probs, Y_val[0])


        axs[1, 0].plot(percentiles1, rates1, marker='.', label=Text_label[idx])
        
        disp1 = CalibrationDisplay.from_estimator(Model, X_val[idx], Y_val[0], label=Text_label[idx], ax=axs[1, 1])
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend() 
    #plt.savefig('AllFigures_15.png')

    plt.show()
    
def plot_ConfusionMatrix(y_pred,y_test,name):
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #plt.savefig(name)
    plt.show()


# ### Question-1:




exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_DistanceFromNet+Angle')




### XGBoost on distance+angle ###

X_train_1 = X_train[['distanceFromNet', 'angle']].to_numpy().reshape(-1, 2)
X_val_1 = x_val[['distanceFromNet', 'angle']].to_numpy().reshape(-1, 2)

xgb_clf_Distance_Angle = XGBClassifier().fit(X_train_1, y_train)
print(type(xgb_clf_Distance_Angle))


train_score1 = xgb_clf_Distance_Angle.score(X_train_1, y_train)
val_score1 = xgb_clf_Distance_Angle.score(X_val_1, y_val)
preds_Val = xgb_clf_Distance_Angle.predict(X_val_1)
plot_ConfusionMatrix(preds_Val,y_val,"Confusion_Matrix_Distance_Angle")

print(classification_report(y_val, preds_Val))
print(f'Training accuracy: {train_score1}')
print(f'Validation accuracy: {val_score1}')
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))

print("brier_score_loss :",brier_score_loss(y_val, preds_Val) )
print("F1_score :",f1_score(y_val, preds_Val,average='weighted') )

exp.log_metrics({'Training accuracy': train_score1, 'Validation accuracy': val_score1})
exp.end()




print(f'Validation accuracy: {val_score1}')
val_score1 = xgb_clf_Distance_Angle.score(X_val_1, y_val)
preds_Val = xgb_clf_Distance_Angle.predict(X_val_1)
plot_ConfusionMatrix(preds_Val,y_val,"Confusion_Matrix_Distance_Angle")

print(classification_report(y_val, preds_Val))
print(f'Training accuracy: {train_score1}')
print(f'Validation accuracy: {val_score1}')
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))


# ### Question-2: Features used for discussion
# Game seconds-
# Game period-
# Coordinates (x,y, separate columns)-
# Shot distance-
# Shot angle-
# Shot type-
# Last event type-
# Coordinates of the last event (x, y, separate columns)-
# Time from the last event (seconds)-
# Distance from the last event-
# Rebound (bool): -
# Change in shot angle-
# “Speed”-

# ### XGBoost using all features only preprocessing  (no feature selection) 





dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(x_val, label=y_val)
exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_AllFeaturesUsed')
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'binary:hinge',
}
num_boost_round = 999
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Validation")],
    early_stopping_rounds=10
)

ax = plot_importance(model)

ax.figure.savefig('FeatureImportance_XGboost_.png')

preds_Val = model.predict(dtest)

acc=accuracy_score(y_val, preds_Val)
print(f'Validation accuracy: ', acc)


plot_ConfusionMatrix(preds_Val,y_val,"Confusion_Matrix_AllFeatures.png")

print(classification_report(y_val, preds_Val))
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))
print("brier_score_loss :",brier_score_loss(y_val, preds_Val) )
print("F1_score :",f1_score(y_val, preds_Val) )
exp.log_metrics({'Training accuracy': train_score1, 'Validation accuracy': val_score1})
exp.log_metrics({'ROC AUC': roc_auc_score(y_val, preds_Val), 'brier_score_loss' :brier_score_loss(y_val, preds_Val)})
exp.end()





### XGBoost using all features  ###

exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_AllFeaturesUsed')


X_train1XG_All = X_train.to_numpy().reshape(-1, len(X_train.columns))
X_val1XG_All = x_val.to_numpy().reshape(-1, len(X_train.columns))

xgb_clf = XGBClassifier().fit(X_train1XG_All, y_train)


ax = plot_importance(xgb_clf)
#ax.figure.savefig('xgb_clf.png')
#pyplot.savefig('n_estimators_vs_max_depth.png')

train_score1 = xgb_clf.score(X_train1XG_All, y_train)
val_score1 = xgb_clf.score(X_val1XG_All, y_val)


print(f'Training accuracy: {train_score1}')
print(f'Validation accuracy: {val_score1}')
preds_Val = xgb_clf.predict(X_val1XG_All)
#plot_ConfusionMatrix(preds_Val,y_val,'Confusion_Matrix_Distance_Angle.png')
print(classification_report(y_val, preds_Val))
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))
print("brier_score_loss :",brier_score_loss(y_val, preds_Val) )
print("F1_score :",f1_score(y_val, preds_Val) )
exp.log_metrics({'Training accuracy': train_score1, 'Validation accuracy': val_score1})
exp.log_metrics({'ROC AUC': roc_auc_score(y_val, preds_Val), 'brier_score_loss' :brier_score_loss(y_val, preds_Val)})
exp.log_metrics({'F1_score': f1_score(y_val, preds_Val)})
exp.end()


# ### Tuning n_estimators; number of Decision Trees (or rounds)  (for evaluation we use f1_Score)





# grid search
exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_Tuning')
model = XGBClassifier()
n_estimators = range(50, 100, 50)
#max_depth = [2, 4, 6, 8]
max_depth = [2]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#f1_score for binary classification
grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=-1, cv=kfold,verbose=1)
grid_result = grid_search.fit(X_train, y_train)
# results
# Mean cross-validated score of the best_estimator


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for i in range(len(GridSearchCV.cv_results_['params'])):

    for k,v in GridSearchCV.cgrid_scores_:
        if k == "params":
            print(v[i])
            exp.log_parameters(v[i])
        else:
            print(k,v[i])
            exp.log_metric(k,v[i])
# plot results
means = grid_result.cv_results_['mean_test_score']
scores = np.array(means).reshape(len(max_depth), len(n_estimators))

for i, value in enumerate(max_depth):
    pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('f1_score')
exp.end()
#pyplot.savefig('n_estimators_vs_max_depth.png')


# ### Question-3: Features selection

# ### Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue





# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(xgb_clf)
shap_values = explainer(X_train)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
print(shap.__version__)





#### Run XGboost with features that are pushing the prediction higher with xgboost:

exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_SHAP_Features')

X_trainXG_SHAP_=X_train[['changeInAngleShot','lastEventXCoord','angle','distanceFromNet']]
X_valXG_SHAP=x_val[['changeInAngleShot','lastEventGameSeconds','angle','distanceFromNet']]

X_trainXG_SHAP = X_trainXG_SHAP_.to_numpy().reshape(-1, len(X_trainXG_SHAP_.columns))
X_val1XG_SHAP = X_valXG_SHAP.to_numpy().reshape(-1, len(X_trainXG_SHAP_.columns))

xgb_cl_SHAP = XGBClassifier(n_estimators=350,max_depth=8).fit(X_trainXG_SHAP, y_train)


plot_importance(xgb_cl_SHAP)


train_score1 = xgb_cl_SHAP.score(X_trainXG_SHAP, y_train)
val_score1 = xgb_cl_SHAP.score(X_val1XG_SHAP, y_val)


print(f'Training accuracy: {train_score1}')
print(f'Validation accuracy: {val_score1}')
preds_Val = xgb_cl_SHAP.predict(X_val1XG_SHAP)
plot_ConfusionMatrix(preds_Val,y_val,"Confusion_Matrix_SelectedFeatures_SHAP.png")
print(classification_report(y_val, preds_Val))
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))
print("brier_score_loss :",brier_score_loss(y_val, preds_Val) )
print("F1_score :",f1_score(y_val, preds_Val,average='weighted') )

exp.log_metrics({'Training accuracy': train_score1, 'Validation accuracy': val_score1})
exp.log_metrics({'ROC AUC': roc_auc_score(y_val, preds_Val), 'brier_score_loss' :brier_score_loss(y_val, preds_Val)})
exp.log_metrics({'F1_score': f1_score(y_val, preds_Val)})
exp.end()


# ## Lasso


clf = linear_model.Lasso(alpha=0.1)
selector = feature_selection.SelectFromModel(estimator=clf,threshold="mean").fit(X_train,y_train)
selected_feature_indices = np.where(selector._get_support_mask())[0] 
res_list = [X_train.columns[i] for i in selected_feature_indices] 
res_list


# ### XGBoost classifier using selected featurs by Lasso: ( we add distanceFromLastEvent) 




### XGBoost using all features  ###


exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_Lasso_Features')
#I choose to  add distanceFromLastEvent feature to include information related to lastEvent 
X_trainXG_select=X_train[['distanceFromNet','distanceFromLastEvent','speed']]
X_valXG_select=x_val[['distanceFromNet','distanceFromLastEvent','speed']]



X_train1XG_select = X_trainXG_select.to_numpy().reshape(-1, len(X_trainXG_select.columns))
X_val1XG_select = X_valXG_select.to_numpy().reshape(-1, len(X_trainXG_select.columns))

xgb_clf_select = XGBClassifier(n_estimators=350,max_depth=8).fit(X_train1XG_select, y_train)


plot_importance(xgb_clf_select)


train_score1 = xgb_clf_select.score(X_train1XG_select, y_train)
val_score1 = xgb_clf_select.score(X_val1XG_select, y_val)

print(f'Training accuracy: {train_score1}')
print(f'Validation accuracy: {val_score1}')
preds_Val = xgb_clf_select.predict(X_val1XG_select)
plot_ConfusionMatrix(preds_Val,y_val,"Confusion_Matrix_SelectedFeaturesLasso.png")
print(classification_report(y_val, preds_Val))
print("ROC AUC : ", roc_auc_score(y_val, preds_Val))
print("brier_score_loss :",brier_score_loss(y_val, preds_Val) )
print("F1_score :",f1_score(y_val, preds_Val,average='weighted') )

exp.log_metrics({'Training accuracy': train_score1, 'Validation accuracy': val_score1})
exp.log_metrics({'ROC AUC': roc_auc_score(y_val, preds_Val), 'brier_score_loss' :brier_score_loss(y_val, preds_Val)})
exp.log_metrics({'F1_score': f1_score(y_val, preds_Val)})
exp.end()




displayFigures([xgb_clf,xgb_cl_SHAP,xgb_clf_select,xgb_clf_Distance_Angle],[X_train1XG_All,X_trainXG_SHAP_,X_train1XG_select,X_train_1],[y_train],[X_val1XG_All,X_val1XG_SHAP,X_val1XG_select,X_val_1],[y_val],["XGBOOST_AllFeatures","xgb_cl_SHAP","LassoSelection","Distance_Angle_Only"])






clf = Pipeline([
  ('feature_selection', SelectFromModel(LogisticRegression())),
  ('classification', XGBClassifier())
])
clf.fit(X_train, y_train)
predictions = clf.predict(x_val)
accuracy = accuracy_score(y_val, predictions)
train_score = clf.score(X_train, y_train)
plot_ConfusionMatrix(predictions,y_val)
print(classification_report(y_val, preds_Val))
print("Training Accuracy: %.1f%%" % ( train_score*100.0))
print(f'Validation accuracy: {accuracy*100.0}')
print("ROC AUC : ", roc_auc_score(y_val, predictions))


# ### Feature Selection using Wrapper methods




# fit model on all training data
exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/XGboost_WrapperMethod')
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
predictions = model.predict(x_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
        #train
    accuracyTrain= selection_model.score(select_X_train, y_train)
        #validation
    select_X_test = selection.transform(x_val)
    predictions = selection_model.predict(select_X_test)
    accuracyValidation = accuracy_score(y_val, predictions)
    print("Thresh=%.3f, n=%d, accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracyTrain*100.0))
    print("Thresh=%.3f, n=%d, Validation: %.2f%%" % (thresh, select_X_train.shape[1], accuracyValidation*100.0))
    print("ROC AUC : ", roc_auc_score(y_val, predictions))
    exp.log_metrics({'Thresh': thresh, 'train accuracy': accuracyTrain*100.0})
    exp.log_metrics({'Thresh': thresh, 'validation accuracy': accuracyValidation*100.0})
    exp.log_metrics({'ROC AUC': roc_auc_score(y_val, predictions)})
   
    exp.end()


# ### Logging the chosen XGBoost Model:




def register_comet_model(workspace, project, experiment, model):
    api = API()
    experiment = api.get(f"{workspace}/{project}/{experiment}")
    experiment.register_model(f"{model}")





exp = Experiment(
    api_key=COMET_API_KEY,
    project_name='ift6758',
    workspace='meriembchaaben',
)
exp.set_name('Question5/SelectedModel')
dump(xgb_clf_Distance_Angle, '../models/XGBoost_Task5.joblib')
exp.log_model("XGBoost (Task5) Model", "../models/XGBoost_Task5.joblib")
exp.end()
register_comet_model('meriembchaaben', 'ift6758', 'Question5/SelectedModel', 'XGBoost (Task5) Model')


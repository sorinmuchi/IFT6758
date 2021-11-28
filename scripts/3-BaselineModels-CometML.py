from comet_ml import Experiment
from comet_ml import API
import os
from dotenv import load_dotenv
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from joblib import dump, load

warnings.filterwarnings("ignore")
load_dotenv('../.env')
COMET_API_KEY = os.getenv('COMET_API_KEY')
COMET_PROJECT_NAME = os.getenv('COMET_PROJECT_NAME')
COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')


# create a comet.ml experiment for dataset stats
exp0 = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)
exp0.set_name('Task-3/dataset-stats')

### Read and preprocess data ###
df = pd.read_csv('../data/trainingSet.csv')
df = df[['distanceFromNet', 'angle', 'Goal']]
df = df.rename({'Goal': 'is_goal', 'distanceFromNet': 'distance'}, axis=1)
df = df.dropna().reset_index(drop=True)
df['is_goal'] = df['is_goal'].astype(np.int64)

### Split data ###
X = df[['distance', 'angle']]
y = df['is_goal'].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

print(f'Dataset size: {len(X)}')
print(f'Training dataset size: {len(X_train)}')
print(f'Validation dataset size: {len(X_val)}')

exp0.log_metrics({'Dataset size': len(X), 'Training dataset size': len(X_train), 'Validation dataset size': len(X_val)})


### dataset stats: goals rate ###
def goals_rate(df):
    nb_goals = len(df[df == 1])
    nb_non_goals = len(df[df == 0])
    goals_rate = nb_goals / (nb_goals + nb_non_goals)
    return goals_rate


gr1 = goals_rate(y)
gr2 = goals_rate(y_train)
gr3 = goals_rate(y_val)

print(f'** ALL ** Goals rate: {gr1} - Non-goals rate: {1 - gr1}')
print(f'** Training ** Goals rate: {gr2} - Non-goals rate: {1 - gr2}')
print(f'** Validation ** Goals rate: {gr3} - Non-goals rate: {1 - gr3}')

exp0.log_metrics({'ALL - Goals rate': gr1, 'ALL - non-goals rate': 1 - gr1})
exp0.log_metrics({'Training - Goals rate': gr2, 'Training - non-goals rate': 1 - gr2})
exp0.log_metrics({'Validation - Goals rate': gr3, 'Validation - non-goals rate': 1 - gr3})

exp0.end()

### Logistic regression on distance ###
exp1 = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)
exp1.set_name('Task-3/LR(distance)')

X_train1 = X_train['distance'].to_numpy().reshape(-1, 1)
X_val1 = X_val['distance'].to_numpy().reshape(-1, 1)

lr_clf1 = LogisticRegression().fit(X_train1, y_train)

train_score1 = lr_clf1.score(X_train1, y_train)
val_score1 = lr_clf1.score(X_val1, y_val)

print('***** Logistic regression (distance) *****')
print(f'Training accuracy: {train_score1}')
print(f'Test accuracy: {val_score1}')

exp1.log_metrics({'Training-accuracy': train_score1, 'Validation-accuracy': val_score1})

dump(lr_clf1, '../models/3-lr_distance.joblib')
exp1.log_model("LR (distance) Model", "../models/3-lr_distance.joblib")


### Logistic regression on angle ###
exp2 = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)
exp2.set_name('Task-3/LR(angle)')

X_train2 = X_train['angle'].to_numpy().reshape(-1, 1)
X_val2 = X_val['angle'].to_numpy().reshape(-1, 1)

lr_clf2 = LogisticRegression().fit(X_train2, y_train)

train_score2 = lr_clf2.score(X_train2, y_train)
val_score2 = lr_clf2.score(X_val2, y_val)

print('***** Logistic regression (angle) *****')
print(f'Training accuracy: {train_score2}')
print(f'Test accuracy: {val_score2}')

exp2.log_metrics({'Training-accuracy': train_score2, 'Validation-accuracy': val_score2})

dump(lr_clf2, '../models/3-lr_angle.joblib')
exp2.log_model("LR (angle) Model", "../models/3-lr_angle.joblib")


### Logistic regression on distance + angle ###
exp3 = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)
exp3.set_name('Task-3/LR(distance+angle)')

X_train3 = X_train.to_numpy().reshape(-1, 2)
X_val3 = X_val.to_numpy().reshape(-1, 2)

lr_clf3 = LogisticRegression().fit(X_train3, y_train)

train_score3 = lr_clf3.score(X_train3, y_train)
val_score3 = lr_clf3.score(X_val3, y_val)

print('***** Logistic regression (distance+angle) *****')
print(f'Training accuracy: {train_score3}')
print(f'Test accuracy: {val_score3}')

exp3.log_metrics({'Training-accuracy': train_score3, 'Validation-accuracy': val_score3})

dump(lr_clf3, '../models/3-lr_distance_angle.joblib')
exp3.log_model("LR (distance+angle) Model", "../models/3-lr_distance_angle.joblib")

### Random baseline ###
random_clf = DummyClassifier(strategy="uniform").fit(X_train3, y_train)

train_score4 = random_clf.score(X_train3, y_train)
val_score4 = random_clf.score(X_val3, y_val)

print('***** Random regression *****')
print(f'Training accuracy: {train_score4}')
print(f'Test accuracy: {val_score4}')

### --PLOT 1-- ROC curve - AUC metric ###
lr_probs1 = lr_clf1.predict_proba(X_val1[:, :])[:, 1]
lr_probs2 = lr_clf2.predict_proba(X_val2[:, :])[:, 1]
lr_probs3 = lr_clf3.predict_proba(X_val3[:, :])[:, 1]
random_probs = random_clf.predict_proba(X_val3[:, :])[:, 1]

lr_auc1 = roc_auc_score(y_val, lr_probs1)
lr_auc2 = roc_auc_score(y_val, lr_probs2)
lr_auc3 = roc_auc_score(y_val, lr_probs3)
random_auc = roc_auc_score(y_val, random_probs)

print('Logistic Regression (trained on distance only): ROC AUC=%.3f' % (lr_auc1))
print('Logistic Regression (trained on angle only): ROC AUC=%.3f' % (lr_auc2))
print('Logistic Regression (trained on both distance and angle): ROC AUC=%.3f' % (lr_auc3))
print('Random: ROC AUC=%.3f' % (random_auc))

exp1.log_metrics({'ROC-AUC': lr_auc1})
exp2.log_metrics({'ROC-AUC': lr_auc2})
exp3.log_metrics({'ROC-AUC': lr_auc3})


lr_fpr1, lr_tpr1, _ = roc_curve(y_val, lr_probs1)
lr_fpr2, lr_tpr2, _ = roc_curve(y_val, lr_probs2)
lr_fpr3, lr_tpr3, _ = roc_curve(y_val, lr_probs3)
random_fpr, random_tpr, _ = roc_curve(y_val, random_probs)

plt.figure(figsize=(6, 6))
plt.plot(lr_fpr1, lr_tpr1, marker='.', label='Logistic Regression (distance)')
plt.plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic Regression (angle)')
plt.plot(lr_fpr3, lr_tpr3, marker='.', label='Logistic Regression (distance + angle)')
plt.plot(random_fpr, random_tpr, linestyle='--', marker='.', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
plt.savefig('../figures/roc.png')


### --PLOT 2-- goal_rate = F(shot probability model percentile) ###
def compute_goal_rate_per_percentile(probs, y):
    percentiles = []
    rates = []

    for i in range(0, 101):
        percentile = np.percentile(probs, i)
        r_percentile = np.float32(round(percentile, 2))
        goals = 0
        no_goals = 0
        for idx, p in enumerate(probs):
            if np.float32(round(p, 2)) == r_percentile:
                if y[idx] == 1:
                    goals += 1
                else:
                    no_goals += 1
        rate = goals * 100 / (goals + no_goals)
        percentiles.append(percentile)
        rates.append(rate)
    return percentiles, rates


percentiles1, rates1 = compute_goal_rate_per_percentile(lr_probs1, y_val)
percentiles2, rates2 = compute_goal_rate_per_percentile(lr_probs2, y_val)
percentiles3, rates3 = compute_goal_rate_per_percentile(lr_probs3, y_val)
percentiles4, rates4 = compute_goal_rate_per_percentile(random_probs, y_val)

plt.figure(figsize=(6, 6))
plt.plot(percentiles1, rates1, marker='.', label='Logistic Regression (distance)')
plt.plot(percentiles2, rates2, marker='.', label='Logistic Regression (angle)')
plt.plot(percentiles3, rates3, marker='.', label='Logistic Regression (distance+angle)')
plt.plot(percentiles4, rates4, marker='.', label='Random')
plt.xlabel('Shot probability model percentile')
plt.ylabel('Goal rate')
plt.legend()
plt.ylim([0, 100])
plt.xlim([0, 1])
plt.title('Goal rate')
plt.show()
plt.savefig('../figures/goal_rate_percentile_1.png')


### --PLOT 3-- cumulative portion of goals = F(shot probability model percentile) ###
def compute_cumulative_propotion_of_goals_per_percentile(probs, y):
    percentiles = []
    cum_rates = []
    cum_rate = 0
    total_goals = sum(y)
    cum_goals = 0
    for i in range(0, 101):
        percentile = np.percentile(probs, i)
        cum_goals = 0
        for idx, p in enumerate(probs):
            if p <= percentile:
                if y[idx] == 1:
                    cum_goals += 1
        cum_rate = cum_goals * 100 / total_goals
        percentiles.append(percentile)
        cum_rates.append(cum_rate)
    return percentiles, cum_rates


percentiles1, rates1 = compute_cumulative_propotion_of_goals_per_percentile(lr_probs1, y_val)
percentiles2, rates2 = compute_cumulative_propotion_of_goals_per_percentile(lr_probs2, y_val)
percentiles3, rates3 = compute_cumulative_propotion_of_goals_per_percentile(lr_probs3, y_val)
percentiles4, rates4 = compute_cumulative_propotion_of_goals_per_percentile(random_probs, y_val)

plt.figure(figsize=(6, 6))
plt.plot(percentiles1, rates1, marker='.', label='Logistic Regression (distance)')
plt.plot(percentiles2, rates2, marker='.', label='Logistic Regression (angle)')
plt.plot(percentiles3, rates3, marker='.', label='Logistic Regression (distance+angle)')
plt.plot(percentiles4, rates4, marker='.', label='Random')
plt.xlabel('Shot probability model percentile')
plt.ylabel('Cumulative proportion of goals')
plt.legend(loc='lower right')
plt.ylim([0, 100])
plt.xlim([0, 1])
plt.title('Cumulative proportion of goals')
plt.show()
plt.savefig('../figures/goal_rate_percentile_2.png')

### --PLOT 4-- Reliability diagram (Calibration curve) ###
fig, ax = plt.subplots(figsize=(6, 6))
disp1 = CalibrationDisplay.from_estimator(lr_clf1, X_val1, y_val, label='Logistic Regression (distance)', marker="P",
                                          ax=ax)
disp2 = CalibrationDisplay.from_estimator(lr_clf2, X_val2, y_val, label='Logistic Regression (angle)', marker="*",
                                          ax=ax)
disp3 = CalibrationDisplay.from_estimator(lr_clf3, X_val3, y_val, label='Logistic Regression (distance+angle)',
                                          marker="h", ax=ax)
disp4 = CalibrationDisplay.from_estimator(random_clf, X_val3, y_val, label='Random', marker="s", ax=ax)
ax.legend(loc='upper left')
plt.show()
plt.savefig('../figures/calibration_diagram.png')



### Evaluation of basic models

print('####### Logistic Regression (distance) #######')
target_names = ['Non-goal', 'Goal']
preds1 = lr_clf1.predict(X_val1)
brier1 = brier_score_loss(y_val, preds1)
print(f'Training accuracy: {train_score1}' )
print(f'Test accuracy: {val_score1}' )
print(f'Brier score: {brier1}' )
print(classification_report(y_val, preds1, target_names=target_names))
exp1.log_metrics({'Brier-score': brier1})


print('####### Logistic Regression (angle) #######')
target_names = ['Non-goal', 'Goal']
preds2 = lr_clf2.predict(X_val2)
brier2 = brier_score_loss(y_val, preds2)
print(f'Training accuracy: {train_score2}' )
print(f'Test accuracy: {val_score2}' )
print(f'Brier score: {brier2}' )
print(classification_report(y_val, preds2, target_names=target_names))
exp2.log_metrics({'Brier-score': brier2})


print('####### Logistic Regression (distance+angle) #######')
target_names = ['Non-goal', 'Goal']
preds3 = lr_clf3.predict(X_val3)
brier3 = brier_score_loss(y_val, preds3)
print(f'Training accuracy: {train_score3}' )
print(f'Test accuracy: {val_score3}' )
print(f'Brier score: {brier3}' )
print(classification_report(y_val, preds3, target_names=target_names))
exp3.log_metrics({'Brier-score': brier3})


# Do all the 3 logistic regression models always predict 0 as it is the dominating label ?!  YES!
print(np.sum([round(p) for p in lr_probs1]), np.sum([round(p) for p in lr_probs2]),
      np.sum([round(p) for p in lr_probs3]))



#### ROC: optimal cutoff point for the different LR classifiers

# Optimal cut-off point in ROC for LR(distance)
fpr1, tpr1, thresholds1 = roc_curve(y_val, lr_probs1)
best_cutoff1 = thresholds1[np.argmax(tpr1 - fpr1)]
print("Threshold value is:", best_cutoff1)
exp1.log_metrics({'best-cutoff': best_cutoff1})


# Optimal cut-off point in ROC for LR(distance)
fpr2, tpr2, thresholds2 = roc_curve(y_val, lr_probs2)
best_cutoff2 = thresholds2[np.argmax(tpr2 - fpr2)]
print("Threshold value is:", best_cutoff2)
exp2.log_metrics({'best-cutoff': best_cutoff2})


# Optimal cut-off point in ROC for LR(distance)
fpr3, tpr3, thresholds3 = roc_curve(y_val, lr_probs3)
best_cutoff3 = thresholds3[np.argmax(tpr3 - fpr3)]
print("Threshold value is:", best_cutoff3)
exp3.log_metrics({'best-cutoff': best_cutoff3})



# Now we have a better performance of our model since
# it is also predicting expected goals instead of always predicting non-goal as before.
print(sum((lr_probs1 >= best_cutoff1).astype(int)))


exp1.end()
exp2.end()
exp3.end()


# Register logged models
def register_comet_model(workspace, project, experiment, model):
    api = API()
    experiment = api.get(f"{workspace}/{project}/{experiment}")
    experiment.register_model(f"{model}")


register_comet_model(COMET_WORKSPACE, COMET_PROJECT_NAME, 'Task-3/LR(distance)', 'LR (distance) Model')
register_comet_model(COMET_WORKSPACE, COMET_PROJECT_NAME, 'Task-3/LR(angle)', 'LR (angle) Model')
register_comet_model(COMET_WORKSPACE, COMET_PROJECT_NAME, 'Task-3/LR(distance+angle)', 'LR (distance+angle) Model')


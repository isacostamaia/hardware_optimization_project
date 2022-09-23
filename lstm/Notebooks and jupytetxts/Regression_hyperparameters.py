# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# +
columns_names = ['model_version', 'batch_size', 'criterion', 'dropout', 'epochs', 'hidden_size', 'learning_rate', 'mean_dtw', 'mean_mixed_err', 'num_layers', 'seq_len', 'start_date']

dataframe = pd.read_csv("My_loop_global_var_and_metrics_2.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

# -

dataframe = dataframe.drop(['start_date'], axis=1)
dataframe

# ## Lasso regression

# +
columns_names = ['batch_size', 'criterion', 'dropout', 'epochs', 'hidden_size', 'learning_rate', 'mean_dtw', 'num_layers', 'seq_len']

reg_frame = dataframe[columns_names]

#we'll start by not including the loss function into the regression model, so we'll do one model at a time
loss = 'nn.MSELoss()'
reg_frame = reg_frame[reg_frame.criterion == loss]
regressors  = reg_frame.drop(['mean_dtw', 'criterion'], axis=1)
X = regressors.values
y = reg_frame.mean_dtw.values
# -

X

# +
lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

# #############################################################################
# Bonus: how much can you trust the selection of alpha?

# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
k_fold = KFold(3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")

plt.show()

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


alpha = 0.00010
# Reconstruction with L1 (Lasso) penalization
# the best value of alpha was determined using cross validation
# with LassoCV
rgr_lasso = Lasso(alpha=alpha)
rgr_lasso.fit(X_train, y_train)
rec_l1 = rgr_lasso.coef_

# -

rec_l1

regressors.columns



from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import mmcv
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as jb
from imblearn.over_sampling import SMOTE



scaler = StandardScaler()



# Loading the dataset
'''
from genre10k import feat, label, label2id

X = feat
X = scaler.fit_transform(X)
y = label


jb.dump(scaler, 'genre10k.scaler')
jb.dump(label2id, 'genre10k.label2id')
'''

X, y = mmcv.load("discog_mfcc_feat.pkl")
X = scaler.fit_transform(X)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:


sm = SMOTE(random_state=42)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

X_train, y_train = sm.fit_resample(X_train, y_train)
__import__('pdb').set_trace()
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000], 'decision_function_shape': ['ovr', 'ovo']},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'decision_function_shape': ['ovr', 'ovo']}]

scores = ['accuracy']


print('########dataset loaded###########')
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(decision_function_shape='ovr', probability=True), tuned_parameters, cv=5,
                       scoring='{}'.format(score), n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    jb.dump(clf, 'genre10k.joblib')
    print()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import plot_confusion_matrix

labels_df = pd.read_csv('../get_dl_labels/labels_test.csv', index_col=0)
data_df = pd.read_csv('summer_hts_data_test.csv')
desc_df = pd.read_csv('desc.csv')

part = pd.merge(data_df, desc_df, how='inner', on='SMILES')
final = pd.merge(labels_df, part, how='inner', on='api')

labels = final.preds
features = final.drop(['fname',
                       'eye_morphology_clean',
                       'api',
                       'preds',
                       'SMILES',
                       'eye_morphology_x',
                       'eye_morphology_y',
                       'certainty'], axis=1)
print('done')

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    random_state=0,
                                                    test_size=0.4)

model = RandomForestClassifier(n_estimators=200,
                              verbose=0,
                              n_jobs=-1,
                              random_state=0)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# score = cross_val_score(model, features, labels, cv=kf, scoring='roc_auc')
score = cross_val_score(model, features, labels, cv=kf, scoring='accuracy')

print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

plot_confusion_matrix(model, X_test, y_test)
plt.show()
from summer_data import SummerData
from fastai.vision.all import *
import cv2
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

summer = SummerData()
df = summer.fastai_df

model = load_learner('../../training/data_processing/trained_model.pkl', cpu=True)

true = []
preds = []
for idx, path in enumerate(tqdm(df['fname'])):
    img = torch.tensor(cv2.imread(path)).cpu()
    true.append(df.eye_morphology_clean.values[idx])
    preds.append(model.predict(img)[0].lower())
df['preds'] = preds
df.to_csv('test.csv')

acc = accuracy_score(true, preds)
print(acc)
print(df.preds.value_counts())
print(df.eye_morphology_clean.value_counts())

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['block', 'needle', 'plate'])
# disp = ConfusionMatrixDisplay(cm, display_labels=['needle', 'plate'])

disp.plot()

plt.savefig('./ext_test_conf_mtrx.png')
plt.show()

print('done')

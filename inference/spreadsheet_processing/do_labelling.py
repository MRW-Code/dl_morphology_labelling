from summer_data import SummerData
from fastai.vision.all import *
import cv2
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

summer = SummerData()
df = summer.fastai_df

model = load_learner('../../training/data_processing/trained_model', cpu=True)

true = []
preds = []
for idx, path in enumerate(tqdm(df['fname'])):
    img = torch.tensor(cv2.imread(path)).cpu()
    true.append(df.eye_morphology_clean.values[idx])
    preds.append(model.predict(img)[0].lower())
df['preds'] = preds

acc = accuracy_score(true, preds)
print(acc)

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('../../data/training_images/Images'))
disp.plot()
plt.show()
# plt.savefig('./code_saves/ext_test_conf_mtrx.png')
print('done')

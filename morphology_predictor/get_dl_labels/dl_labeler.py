from inference.spreadsheet_processing.summer_data import SummerData
from fastai.vision.all import *
import os
from tqdm import tqdm
import cv2
import pandas as pd

class DLMorphologyLabeler:


    def __init__(self, retrain=None):
        self.retrain = retrain if retrain is not None else False
        self.path = Path(os.path.dirname('../../'))

    def train_model(self):
        if self.retrain == False:
            pass
        else:
            Path.BASE_PATH = self.path

            crystals = DataBlock(
                blocks=(ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(valid_pct=0.1, seed=0),
                get_y=parent_label)

            dls = crystals.dataloaders(self.path / 'data/training_images/Images', bs=8)

            learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
            learn.fine_tune(3)

            learn.export(self.path / 'training/data_processing/trained_model.pkl')

    def get_labels(self):
        summer = SummerData()
        df = summer.fastai_df

        model = load_learner(self.path / 'training/data_processing/trained_model.pkl', cpu=True)

        true = []
        preds = []
        for idx, path in enumerate(tqdm(df['fname'])):
            img = torch.tensor(cv2.imread(path)).cpu()
            true.append(df.eye_morphology_clean.values[idx])
            preds.append(model.predict(img)[0].lower())
        df['preds'] = preds
        return df

print(os.getcwd())
t = DLMorphologyLabeler()
test = t.get_labels()
test.to_csv('./labels_test.csv')
print(test.head())
print('done')
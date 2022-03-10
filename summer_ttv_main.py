import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from fastai.vision.all import *
import pandas as pd
from sklearn.model_selection import KFold
import random
from sklearn.metrics import accuracy_score
import cv2
import torch
import numpy as np

path = Path(os.path.dirname('./'))
Path.BASE_PATH = path

def get_df():
    ref = pd.read_csv('./data/summer_hts_data_matt.csv', index_col=0)
    fnames = []
    labels = []
    api = []
    for drug in ref.api:
        parent_name = f'./data/water/water/{drug}'
        parent_content = os.listdir(parent_name)
        for name in parent_content:
            api.append(drug)
            fnames.append(f'{parent_name}/{name}')
            labels.append(ref.eye_morphology[ref['api'] == drug].values[0])
    df = pd.DataFrame({'api': api,
                       'fname': fnames,
                       'label': labels})
    return df

def train_model(model_df, num_model):
    # print(model_df.head(20), model_df.shape)
    tfms_item = Resize((480, 640))
    tfms_bat = aug_transforms(pad_mode='zeros', mult=2, min_scale=0.5)
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=1,
                                   label_col=2,
                                   valid_pct=0.2,
                                   item_tfms=tfms_item,
                                   batch_tfms=tfms_bat,
                                   bs=32)

    learn = cnn_learner(dls, resnet18, metrics=[error_rate, accuracy])
    learn.fine_tune(50, cbs=[SaveModelCallback(fname=f'./best_cbs_{num_model}'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2)])
    learn.export(f'./trained_model_val_test_{num_model}.pkl')

    # learn.recorder.plot_loss()
    # plt.savefig(f'./training_plot_val_test_{num_model}.png')

    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # plt.savefig(f'./conf_mtrx_val_test_{num_model}')

if __name__ == '__main__':
    ref_df = get_df()
    api_list = ref_df.api.value_counts().index
    kfold = KFold(n_splits=25)

    split_idx = 0
    final_acc = []
    for train, test in kfold.split(api_list):
        train_df = ref_df[ref_df['api'].isin(api_list[train])]
        test_df = ref_df[ref_df['api'].isin(api_list[test])]

        trainer = train_model(train_df, split_idx)

        model = load_learner(f'./trained_model_val_test_{split_idx}.pkl', cpu=False)
        true = []
        preds = []
        for idx, path in enumerate(test_df['fname']):
            img = torch.tensor(cv2.imread(path)).cpu()
            true.append(test_df.label.values[idx])
            preds.append(model.predict(img)[0].lower())
        acc = accuracy_score(true, preds)
        final_acc.append(acc)
        split_idx = split_idx + 1
        torch.cuda.empty_cache()

    print(final_acc)
    print(f'Mean Test Acc = {np.mean(final_acc)}')
    print('done')


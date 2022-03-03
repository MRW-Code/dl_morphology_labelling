import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from fastai.vision.all import *
import pandas as pd
path = Path(os.path.dirname('./'))
Path.BASE_PATH = path


def get_train_images(path):
    images = get_image_files(path)
    labels = [x.parent.name for x in images]
    df = pd.DataFrame({'fname' : images,
                       'label' : labels})
    df['is_valid'] = 0
    return df

def get_val_labels(path):
    ref = pd.read_csv('./data/summer_hts_data_matt.csv', index_col=0)
    fnames = []
    labels = []
    for drug in ref.api:
        parent_name = f'./data/water/water/{drug}'
        parent_content = os.listdir(parent_name)
        for name in parent_content:
            fnames.append(f'{parent_name}/{name}')
            labels.append(ref.eye_morphology[ref['api'] == drug].values[0])
    df = pd.DataFrame({'fname' : fnames,
                       'label' : labels})
    df['is_valid'] = 1
    return df

# train_df = get_train_images('./data/training_images/Images')
# val_df = get_val_labels('./data/water')
# model_df = pd.concat([train_df, val_df], axis=0)

val_df = get_val_labels('./data/water')

print(model_df.head(), model_df.shape)
tfms = None
dls = ImageDataLoaders.from_df(model_df,
                               fn_col=0,
                               label_col=1,
                               val_pct=0.33,
                               batch_tfms=tfms,
                               bs=32)

learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
learn.fine_tune(20, cbs=[SaveModelCallback(fname='./best_cbs_100'),
                                  ReduceLROnPlateau(monitor='valid_loss',
                                                    min_delta=0.1,
                                                    patience=2)])
learn.export('./trained_model.pkl')



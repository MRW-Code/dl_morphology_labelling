import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from fastai.vision.all import *

path = Path(os.path.dirname('../../../dl_morphology_labelling/'))
Path.BASE_PATH = path


crystals = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter= RandomSplitter(valid_pct=0.1, seed=0),
    get_y =parent_label)

# tfms = aug_transforms(pad_mode='zeros', mult=2, min_scale=0.5)
tfms = aug_transforms(mult=10, do_flip=True,
    flip_vert=False,
    max_rotate=10.0,
    min_zoom=1.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75,
    p_lighting=0.75,
    xtra_tfms=None,
    size=None,
    mode='bilinear',
    pad_mode='zeros',
    align_corners=True,
    batch=False,
    min_scale=1.0)


dls = crystals.dataloaders(path / 'data/training_images/Images', bs=32, batch_tfms=tfms)

learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
learn.fine_tune(20, cbs=[SaveModelCallback(fname='./best_cbs_100'),
                                  ReduceLROnPlateau(monitor='valid_loss',
                                                    min_delta=0.1,
                                                    patience=2)])

learn.export('./trained_model.pkl')

learn.recorder.plot_loss()
plt.savefig('./training_plot.png')
# print(path.ls())


from fastai.vision.all import *
import os

path = Path(os.path.dirname('../../../dl_morphology_labelling/'))
Path.BASE_PATH = path


crystals = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter= RandomSplitter(valid_pct=0.1, seed=0),
    get_y =parent_label)

dls = crystals.dataloaders(path / 'data/training_images/Images', bs=8)

learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
learn.fine_tune(3)

learn.export('./trained_model.pkl')

# print(path.ls())


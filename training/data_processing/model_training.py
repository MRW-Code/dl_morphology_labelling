from fastai.vision.all import *
import os

path = Path(os.path.dirname('../../../dl_morphology_labelling/'))
Path.BASE_PATH = path


crystals = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter= RandomSplitter(valid_pct=0.3, seed=0),
    get_y =parent_label)

crystals.dataloaders(path / 'data/training_images', bs=16)

learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
learn.fine_tune(10)

learn.export('./trained_model')

# print(path.ls())
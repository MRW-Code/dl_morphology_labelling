from unsupervised.encoding import *
from unsupervised.dataset import MorphologyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

def get_image_paths(img_dir):
    img_pths = []
    for folder in os.listdir(img_dir):
        direc = f'{img_dir}{folder}'
        for file in os.listdir(direc):
            path = f'{direc}/{file}'
            img_pths.append(path)
    return img_pths

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCH = 10
model = AutoEncoderVGG().to(DEVICE)
OPT = torch.optim.SGD(model.parameters(), lr=0.001)
LOSS = torch.nn.CrossEntropyLoss()
VAL_SPLIT = 0.2
shuffle_dataset = True
BATCH_SIZE = 32


tfms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,640)),
    transforms.ToTensor()
])
img_dir = './data/training_images/Images/'
img_pths = get_image_paths(img_dir)

dataset = MorphologyDataset(img_pths, transform=tfms)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(VAL_SPLIT * dataset_size))
if shuffle_dataset :
    np.random.seed(42)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                sampler=valid_sampler)

dataloaders = {'train': train_loader,
                    'test': validation_loader}
dataset_sizes = {'train': len(train_loader), 'test': len(validation_loader)}



for epoch in range(N_EPOCH):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        n_instances = 0

        for input in dataloaders[phase]:
            input.to(DEVICE)

            OPT.zero_grad()

            output = model(input)
            loss = LOSS(output, input)

            if phase == 'train':
                loss.backward()
                OPT.step()

            running_loss += loss.item() * input.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
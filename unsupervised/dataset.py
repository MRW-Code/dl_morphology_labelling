from torch.utils.data import Dataset
from torchvision.io import read_image
import PIL

class MorphologyDataset(Dataset):

    def __init__(self, img_pths, transform):
        self.img_pths = img_pths
        self.transform = transform

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        img = read_image(self.img_pths[idx]) / 255
        if self.transform:
            self.transform(img)
        return img

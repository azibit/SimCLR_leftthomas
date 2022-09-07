from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

from torch.utils.data import Dataset
import glob as glob
import os, cv2, random, torch


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class EndoscopyDataset(Dataset):
    def __init__(self, root_data, transform=None, no_of_views=2):
        self.root_data = root_data
        self.file_list = glob.glob(f"{root_data}/images/*")
        self.transform = transform
        self.no_of_views = no_of_views

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.file_list[idx])
        orig_image = cv2.cvtColor(cv2.imread(img_loc), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            pos_1 = self.transform(transforms.ToPILImage()(orig_image))
            pos_2 = self.transform(transforms.ToPILImage()(orig_image))

        return pos_1, pos_2, idx

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class HiraganaDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.img_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.chart = {'a': 0, 'ba': 1, 'be': 2, 'bi': 3, 'bo': 4, 'bu': 5, 'chi': 6, 'da': 7, 'de': 8, 'do': 9, 'e': 10, 'fu': 11, 'ga': 12, 'ge': 13, 'gi': 14, 'go': 15, 'gu': 16, 'ha': 17, 'he': 18, 'hi': 19, 'ho': 20, 'i': 21, 'ji': 22, 'ka': 23, 'ke': 24, 'ki': 25, 'ko': 26, 'ku': 27, 'ma': 28, 'me': 29, 'mi': 30, 'mo': 31, 'mu': 32, 'n': 33, 'na': 34,
                      'ne': 35, 'ni': 36, 'no': 37, 'nu': 38, 'o': 39, 'pa': 40, 'pe': 41, 'pi': 42, 'po': 43, 'pu': 44, 'ra': 45, 're': 46, 'ri': 47, 'ro': 48, 'ru': 49, 'sa': 50, 'se': 51, 'shi': 52, 'so': 53, 'su': 54, 'ta': 55, 'te': 56, 'to': 57, 'tsu': 58, 'u': 59, 'wa': 60, 'wo': 61, 'ya': 62, 'yo': 63, 'yu': 64, 'za': 65, 'ze': 66, 'zo': 67, 'zu': 68}

    def __len__(self):
        return 13800

    def __getitem__(self, index):
        index += 1
        hira_index = ((index-1)//200)
        img_index = ((index-1)%200)
        hira_dir = f"./data/HiraganaDataset/train/hiragana_{list(self.chart.keys())[hira_index]}"
        img = Image.open(f"{hira_dir}/{os.listdir(hira_dir)[img_index]}")
        lbl = hira_index
        return (self.img_to_tensor(img), lbl)

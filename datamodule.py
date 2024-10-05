from torch.utils.data import DataLoader, Dataset
import lightning as L
from transformers import MarianTokenizer
from datasets import load_dataset

from dataset import WMT

class WMT14DataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
    
    def setup(self, stage=None):
        dataset = load_dataset('wmt14', 'de-en')
        self.train_dataset = WMT(data = dataset['train'])
        self.val_dataset = WMT(data = dataset['validation'])
        self.test_dataset = WMT(data = dataset['test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)
    
class WMT(Dataset):
    def __init__(self, data):
        self.data = data
                
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]['translation']['en'], self.data[idx]['translation']['de']
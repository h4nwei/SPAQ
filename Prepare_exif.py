import pandas as pd
import torch

class Exif_load(object):
    def __call__(self, exif_file):
        self.data = pd.read_csv(exif_file, sep='\t', header=None)
        exif_tags = self.normalization_exif(torch.FloatTensor(self.data.iloc[0, 1:9].tolist()))
        return exif_tags

    def normalization_exif(self, x):
        return (x - 3.3985) / 5.6570




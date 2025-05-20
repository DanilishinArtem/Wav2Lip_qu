import torch
from torch.quantization import quantize_dynamic

class Compressor():
    def __init__(self, model):
        self.model = model
        self.get_info()

    def get_info(self):
        for name, layer in self.model.named_modules():
            print("layer: {}".format(name))


    def compress(self):
        quantize_dynamic(self.model, inplace=True)
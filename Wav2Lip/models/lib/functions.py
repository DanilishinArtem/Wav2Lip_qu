
import torch
import torch.nn as nn
from lib.config import Config

if Config.quant_method == 'pca':
    from lib.quantizator_pca import QuantizedConv2d, QuantizedConvTranspose2d
else:
    from lib.quantizator_int8_to_fp16 import QuantizedConv2d, QuantizedConvTranspose2d

from wav2lip import Wav2Lip
import torch.onnx
import os

total_number = 0
changed = 0
def quantize_model(model):
    global total_number
    global changed
    total_number += 1
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            changed += 1
            setattr(model, name, QuantizedConv2d(child))
        elif isinstance(child, nn.ConvTranspose2d):
            changed += 1
            setattr(model, name, QuantizedConvTranspose2d(child))
        else:
            quantize_model(child)


def get_model_data():
    global total_number
    global changed
    model = Wav2Lip()
    mel = torch.rand(Config.batch_size, 1, 80, 16)
    face = torch.rand(Config.batch_size, 6, 96, 96)
    if Config.half:
        model = model.half()
        mel = mel.half()
        face = face.half()
    model = model.to(Config.device)
    mel = mel.to(Config.device)
    face = face.to(Config.device)
    if Config.quant:
        print("Quantizing model...")
        quantize_model(model)
        print("{} quantized layers out of {}".format(changed, total_number))
    model.eval()
    return model, mel, face


def export_onnx(model, mel, face, name):
    os.system("rm -rf /home/adanilishin/wav2lip/Wav2Lip/models/lib/{}.onnx".format(name))
    torch.onnx.export(
        model,
        (mel, face),
        "{}.onnx".format(name),
        opset_version=11,
        input_names=['mel', 'face'],
        output_names=['output'],
        dynamic_axes={
            'mel': {0: 'batch_size'},
            'face': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

def one_step(model, mel, face):
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(Config.device)
        out = model(mel, face)
        print("Peak memory (MB):", torch.cuda.max_memory_allocated(Config.device) / 1024**2)
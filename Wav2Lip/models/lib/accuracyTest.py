import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from wav2lip import Wav2Lip
import copy
from lib.functions import quantize_model, get_model_data
from lib.config import Config
import os


def compare_quantization(name):
    os.system("rm -rf /home/adanilishin/wav2lip/Wav2Lip/models/lib/quantization_comparison.png")
    Config.quant = False
    model_orig, mel_input, face_input = get_model_data()
    model_orig = model_orig.to('cpu')
    model_quant = copy.deepcopy(model_orig)

    models = {
        "Original": model_orig,
        "Quantized": model_quant
    }
    
    quantize_model(models["Quantized"])
    
    if Config.half:
        for name, m in models.items():
            m = m.half()
    
    results = {}
    for name, m in models.items():
        m.eval()
        m.to(Config.device)
        with torch.no_grad():
            # Перенос входных данных с учетом типа
            face = face_input.clone().to(Config.device)
            mel = mel_input.clone().to(Config.device)
            
            if Config.half:
                face = face.half()
                mel = mel.half()
                
            output = m(mel, face)
            results[name] = output.detach().cpu()
    
    orig_tensor = results["Original"].float()
    quant_tensor = results["Quantized"].float()
    
    diff = {
        "MSE": torch.mean((orig_tensor - quant_tensor) ** 2).item(),
        "MAE": torch.mean(torch.abs(orig_tensor - quant_tensor)).item(),
        "Max Error": torch.max(torch.abs(orig_tensor - quant_tensor)).item(),
        "Min Error": torch.min(torch.abs(orig_tensor - quant_tensor)).item(),
        "PSNR": 10 * torch.log10(1 / torch.mean((orig_tensor - quant_tensor) ** 2)).item()
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    orig_img = make_grid(orig_tensor[0], nrow=1, normalize=True, value_range=(-1, 1)).permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(orig_img, 0, 1))
    axes[0].set_title("Original Output")
    axes[0].axis('off')
    
    quant_img = make_grid(quant_tensor[0], nrow=1, normalize=True, value_range=(-1, 1)).permute(1, 2, 0).numpy()
    axes[1].imshow(np.clip(quant_img, 0, 1))
    axes[1].set_title("Quantized Output")
    axes[1].axis('off')
    
    diff_img = make_grid(torch.abs(orig_tensor - quant_tensor)[0], nrow=1, normalize=True).permute(1, 2, 0).numpy()
    axes[2].imshow(diff_img, cmap='hot')
    axes[2].set_title("Difference (Enhanced)")
    axes[2].axis('off')
    
    metrics_text = "\n".join([f"{k}: {v:.6f}" for k, v in diff.items()])
    plt.figtext(0.5, 0.05, metrics_text, ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.suptitle("Quantization Comparison", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("{}_{}.png".format(name, Config.quant_method), dpi=300, bbox_inches="tight", pad_inches=0.1)

    # Вывод метрик
    print("\nQuantization Metrics:")
    for metric, value in diff.items():
        print(f"{metric}: {value:.6f}")
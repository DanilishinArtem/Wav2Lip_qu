from lib.functions import *
from lib.config import Config
from lib.accuracyTest import compare_quantization

if __name__ == "__main__":
    name = None
    if Config.quant:
        name = "quant"
    else:
        name = "baseline"

    print("[DEBUG] GET MODEL DATA...")
    model, mel, face = get_model_data()
    print("[DEBUG] END OF GETTING MODEL DATA...")
    if Config.export_onnx:
        print("----------------- EXPORT ONNX -----------------")
        export_onnx(model, mel, face, name)
        x = os.path.getsize("/home/adanilishin/wav2lip/Wav2Lip/models/lib/{}.onnx".format(name))
        x = x / 1024 / 1024
        print("Size of {}.onnx = {} mb".format(name, x))
    if Config.step:
        print("----------------- ONE STEP -----------------")
        one_step(model, mel, face)
    if Config.accuracy_test:
        print("----------------- ACCURACY TEST -----------------")
        compare_quantization(name)
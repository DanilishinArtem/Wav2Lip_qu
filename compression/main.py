from Wav2Lip.models import Wav2Lip
from compression.compressor import Compressor
import torch

def get_model():
    model = Wav2Lip()
    model.eval()
    return model

if __name__ == "__main__":
    model = get_model()
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    # Compressor(model)
    state_dict = torch.load("...")
    # print(model)
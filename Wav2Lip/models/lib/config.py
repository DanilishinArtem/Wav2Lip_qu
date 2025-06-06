class Config:
    device = 'cuda'
    quant = False
    # pca / not_pca
    quant_method = 'pca'
    half = True
    batch_size = 1
    pca_accuracy = 0.5
    step = False
    export_onnx = True
    accuracy_test = False
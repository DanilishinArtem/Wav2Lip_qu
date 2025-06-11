class Config:
    device = 'cuda'
    quant = True
    # pca / not_pca
    quant_method = 'not_pca'
    half = True
    batch_size = 1
    pca_accuracy = 0.5
    step = True
    export_onnx = False
    accuracy_test = False
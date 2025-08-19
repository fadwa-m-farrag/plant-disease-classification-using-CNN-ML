import torch
from cnn import CNN

device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load("cnn_feature_extractor.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1,3,256,256)

torch.onnx.export(
    model, 
    dummy_input,
    "cnn.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['features']
)
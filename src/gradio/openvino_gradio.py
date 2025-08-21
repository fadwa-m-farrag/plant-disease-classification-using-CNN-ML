import gradio as gr
from openvino.runtime import Core
from PIL import Image
from torchvision.transforms import v2
import joblib
import torch

ie = Core()
model_ir = ie.read_model(model="cnn.xml")
compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

pca = joblib.load("model files/pca.pkl")
clf = joblib.load("model files/xgb.pkl")
class_names = joblib.load("model files/classNames.pkl")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def predict_image(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).numpy()

    features = compiled_model([img_tensor])[output_layer]

    features_pca = pca.transform(features)
    prediction_idx = clf.predict(features_pca)[0]

    return class_names[prediction_idx]

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Apple Plant Disease Classification (OpenVINO)",
    description="Upload a leaf image to classify its disease type."
)

if __name__ == "__main__":
    interface.launch()

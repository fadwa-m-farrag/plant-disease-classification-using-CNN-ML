import gradio as gr
import torch
import joblib
from PIL import Image
from torchvision.transforms import v2
from cnn import CNN

device = torch.device("cuda")
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_feature_extractor.pth", map_location=device))
model.eval()

pca = joblib.load("pca_model.pkl")
clf = joblib.load("ml_classifier.pkl")
class_names = joblib.load("class_names.pkl")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def predict(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor).cpu().numpy()

    features = pca.transform(features)

    pred = clf.predict(features)[0]
    return class_names[pred]

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a apple leaf image"),
    outputs=gr.Label(label="Predicted Class"),
    title="Apple Plant Leaf Disease Detection",
    description="Upload an apple plant leaf image to detect the disease class."
)

interface.launch()

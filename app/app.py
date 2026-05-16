import streamlit as st
import torch
import timm
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import tempfile
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- FIX RANDOMNESS ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

st.set_page_config(page_title="Deepfake Detection", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():

    efficient_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    efficient_model.load_state_dict(torch.load("efficientnet_faces_fina2.pth", map_location=device))
    efficient_model.to(device).eval()

    xception_model = timm.create_model("xception", pretrained=False, num_classes=2)
    xception_model.load_state_dict(torch.load("xception_deepfake_final.pth", map_location=device))
    xception_model.to(device).eval()

    return efficient_model, xception_model

efficient_model, xception_model = load_models()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- FACE DETECTOR ----------------
face_detector = MTCNN(image_size=160, margin=10, device=device)

# ---------------- FACE DETECTION ----------------
def detect_face(image):

    boxes, _ = face_detector.detect(image)

    if boxes is None:
        return None

    x1,y1,x2,y2 = boxes[0].astype(int)

    face = image.crop((x1,y1,x2,y2))

    return face

# ---------------- PREDICTION ----------------
def predict(model,image):

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output,dim=1)

    return prob[0][0].item(), prob[0][1].item()

# ---------------- GRAD CAM ----------------
def generate_gradcam(model, face_img):

    input_tensor = transform(face_img).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=[model.conv_head])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    face_np = np.array(face_img.resize((160,160))) / 255.0

    visualization = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)

    return visualization

# ---------------- UI ----------------
st.title("Deepfake Detection System")

model_choice = st.selectbox("Select Model", ["XceptionNet","EfficientNet","Grad-CAM"])

if model_choice == "Grad-CAM":
    detection_type = "Image"
else:
    detection_type = st.selectbox("Detection Type", ["Image","Video"])

st.divider()

# ---------------- IMAGE ----------------
if detection_type == "Image":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:

        image = Image.open(file).convert("RGB")

        st.image(image, width=400)

        face = detect_face(image)

        if face is None:
            st.warning("No face detected")
            st.stop()

        if st.button("Run"):

            if model_choice == "Grad-CAM":
                cam_img = generate_gradcam(efficient_model, face)
                st.image(cam_img, width=400)

            else:
                model = efficient_model if model_choice=="EfficientNet" else xception_model
                fake, real = predict(model, face)

                st.metric("Fake", round(fake,3))
                st.metric("Real", round(real,3))

                if fake > real:
                    st.error("Fake Image")
                else:
                    st.success("Real Image")

# ---------------- VIDEO ----------------
if detection_type == "Video":

    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:

        st.warning("Video mode requires OpenCV → not supported in deployment")

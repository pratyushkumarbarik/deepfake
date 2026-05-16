import streamlit as st
import torch
import timm
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import matplotlib.cm as cm
import gdown
import os
import tempfile
import imageio.v2 as imageio   # ✅ IMPORTANT FIX

# ---------------- FIX RANDOMNESS ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

st.set_page_config(page_title="Deepfake Detection", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DOWNLOAD + LOAD MODELS ----------------
@st.cache_resource
def load_models():

    if not os.path.exists("efficientnet.pth"):
        gdown.download(
            "https://drive.google.com/uc?id=1uufNBM-cjvQFRDA1eber_FYqb_bBRV4C",
            "efficientnet.pth",
            quiet=False
        )

    if not os.path.exists("xception.pth"):
        gdown.download(
            "https://drive.google.com/uc?id=1ZlvAT2nfqPgeyhHP37OcNf8s68uvjD7d",
            "xception.pth",
            quiet=False
        )

    efficient_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    efficient_model.load_state_dict(torch.load("efficientnet.pth", map_location=device))
    efficient_model.to(device).eval()

    xception_model = timm.create_model("xception", pretrained=False, num_classes=2)
    xception_model.load_state_dict(torch.load("xception.pth", map_location=device))
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
    return image.crop((x1,y1,x2,y2))

# ---------------- PREDICTION ----------------
def predict(model,image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output,dim=1)
    return prob[0][0].item(), prob[0][1].item()

# ---------------- GRAD-CAM ----------------
def generate_gradcam(model, image):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.conv_head

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)

    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    # Heatmap fix
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((160,160))
    ) / 255.0

    img_np = np.array(image.resize((160,160))) / 255.0

    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay

# ---------------- VIDEO FRAME EXTRACTION ----------------
def extract_frames(video_path, num_frames=20):

    reader = imageio.get_reader(video_path, "ffmpeg")

    try:
        total = reader.count_frames()
    except:
        total = 100  # fallback

    ids = np.linspace(0, total-1, num_frames).astype(int)

    frames = []

    for i in ids:
        try:
            frame = reader.get_data(i)
            frames.append(Image.fromarray(frame))
        except:
            continue

    reader.close()
    return frames

# ---------------- UI ----------------
st.title("Deepfake Detection System")

model_choice = st.selectbox(
    "Select Mode",
    ["EfficientNet","XceptionNet","Grad-CAM"]
)

if model_choice != "Grad-CAM":
    detection_type = st.selectbox("Detection Type", ["Image","Video"])
else:
    detection_type = "Image"

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

                st.subheader("Grad-CAM Visualization")
                st.image(cam_img, width=400)

            else:

                model = efficient_model if model_choice=="EfficientNet" else xception_model

                fake, real = predict(model, face)

                st.subheader("Prediction")

                c1,c2 = st.columns(2)
                c1.metric("Fake Probability", round(fake,3))
                c2.metric("Real Probability", round(real,3))

                if fake > real:
                    st.error("Fake Image")
                else:
                    st.success("Real Image")

# ---------------- VIDEO ----------------
if detection_type == "Video":

    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        st.video(tfile.name)

        if st.button("Detect Video"):

            frames = extract_frames(tfile.name)

            fake_scores = []
            real_scores = []

            for frame in frames:

                face = detect_face(frame)

                if face is None:
                    continue

                model = efficient_model if model_choice=="EfficientNet" else xception_model

                fake, real = predict(model, face)

                fake_scores.append(fake)
                real_scores.append(real)

            if len(fake_scores)==0:
                st.warning("No faces detected")
                st.stop()

            fake_avg = np.mean(fake_scores)
            real_avg = np.mean(real_scores)

            st.subheader("Prediction")

            c1,c2 = st.columns(2)
            c1.metric("Fake Probability", round(fake_avg,3))
            c2.metric("Real Probability", round(real_avg,3))

            if fake_avg > real_avg:
                st.error("Fake Video")
            else:
                st.success("Real Video")

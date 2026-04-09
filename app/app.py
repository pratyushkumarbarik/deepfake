import streamlit as st
import torch
import timm
import cv2
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

    efficient_model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    efficient_model.load_state_dict(
        torch.load(
            "/mnt/d/deepfake/tf-gpu-env/complete_project/models/efficientnet_faces_fina2.pth",
            map_location=device
        )
    )

    efficient_model.to(device)
    efficient_model.eval()

    xception_model = timm.create_model(
        "xception",
        pretrained=False,
        num_classes=2
    )

    xception_model.load_state_dict(
        torch.load(
            "/mnt/d/deepfake/tf-gpu-env/complete_project/models/xception_deepfake_final.pth",
            map_location=device
        )
    )

    xception_model.to(device)
    xception_model.eval()

    return efficient_model, xception_model


efficient_model, xception_model = load_models()

# ---------------- IMAGE TRANSFORM (MATCH JUPYTER) ----------------
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- FACE DETECTOR ----------------
face_detector = MTCNN(
    image_size=160,
    margin=10,
    device=device
)

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, num_frames=50):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= num_frames:
        ids = np.arange(total)
    else:
        ids = np.linspace(0,total-1,num_frames).astype(int)

    frames = []

    for i in ids:

        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    cap.release()

    return frames


# ---------------- FACE DETECTION ----------------
def detect_face(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)

    boxes,_ = face_detector.detect(img)

    if boxes is None:
        return None

    x1,y1,x2,y2 = boxes[0].astype(int)

    face = frame[y1:y2,x1:x2]

    if face.size == 0:
        return None

    return face


# ---------------- PREDICTION ----------------
def predict(model,image):

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        prob = torch.softmax(output,dim=1)

    fake = prob[0][0].item()
    real = prob[0][1].item()

    return fake,real


# ---------------- GRAD CAM (IDENTICAL TO JUPYTER) ----------------
def generate_gradcam(model, face_img):

    model.eval()

    input_tensor = transform(face_img).unsqueeze(0).to(device)

    target_layer = model.conv_head

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    # SAME AS JUPYTER
    grayscale_cam = cam(input_tensor=input_tensor)

    heatmap = grayscale_cam[0]

    face_resized = face_img.resize((160,160))

    face_np = np.array(face_resized) / 255.0

    visualization = show_cam_on_image(
        face_np,
        heatmap,
        use_rgb=True,
        image_weight=0.7
    )

    return visualization


# ---------------- STREAMLIT UI ----------------
st.title("Deepfake Detection System")

model_choice = st.selectbox(
    "Select Model",
    ["XceptionNet","EfficientNet","Grad-CAM"]
)

if model_choice == "Grad-CAM":
    detection_type = "Image"
else:
    detection_type = st.selectbox(
        "Detection Type",
        ["Image","Video"]
    )

st.divider()

# ---------------- IMAGE MODE ----------------
if detection_type == "Image":

    file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if file:

        original = Image.open(file).convert("RGB")

        st.image(original,width=450)

        frame = cv2.cvtColor(np.array(original),cv2.COLOR_RGB2BGR)

        face = detect_face(frame)

        if face is None:
            st.warning("No face detected")
            st.stop()

        face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

        face_img = Image.fromarray(face_rgb)

        if st.button("Run"):

            if model_choice == "Grad-CAM":

                cam_img = generate_gradcam(efficient_model, face_img)

                st.subheader("Grad-CAM Visualization")

                st.image(cam_img,width=400)

            else:

                if model_choice == "EfficientNet":
                    fake,real = predict(efficient_model,face_img)
                else:
                    fake,real = predict(xception_model,face_img)

                st.subheader("Prediction")

                c1,c2 = st.columns(2)

                c1.metric("Fake Probability",round(fake,3))
                c2.metric("Real Probability",round(real,3))

                if fake > real:
                    st.error("Fake Image")
                else:
                    st.success("Real Image")


# ---------------- VIDEO MODE ----------------
if detection_type == "Video":

    file = st.file_uploader("Upload Video",type=["mp4","avi","mov"])

    if file:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(file.read())

        st.video(tfile.name)

        if st.button("Detect Video"):

            frames = extract_frames(tfile.name)

            fake_scores=[]
            real_scores=[]

            for frame in frames:

                face = detect_face(frame)

                if face is None:
                    continue

                face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

                face_img = Image.fromarray(face_rgb)

                if model_choice == "EfficientNet":
                    fake,real = predict(efficient_model,face_img)
                else:
                    fake,real = predict(xception_model,face_img)

                fake_scores.append(fake)
                real_scores.append(real)

            if len(fake_scores)==0:

                st.warning("No usable faces detected")
                st.stop()

            fake_avg = np.mean(fake_scores)
            real_avg = np.mean(real_scores)

            st.subheader("Prediction")

            c1,c2 = st.columns(2)

            c1.metric("Fake Probability",round(fake_avg,3))
            c2.metric("Real Probability",round(real_avg,3))

            if fake_avg > real_avg:
                st.error("Fake Video")
            else:
                st.success("Real Video")
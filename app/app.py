import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile
from facenet_pytorch import MTCNN

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

    efficient_model = efficient_model.to(device)
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

    xception_model = xception_model.to(device)
    xception_model.eval()

    return efficient_model, xception_model


efficient_model, xception_model = load_models()


# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---------------- FACE DETECTOR ----------------

face_detector = MTCNN(
    keep_all=True,
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

    # choose largest face
    areas = []
    for box in boxes:
        x1,y1,x2,y2 = box
        areas.append((x2-x1)*(y2-y1))

    idx = np.argmax(areas)

    x1,y1,x2,y2 = boxes[idx].astype(int)

    h,w,_ = frame.shape

    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(w,x2)
    y2 = min(h,y2)

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


# ---------------- STREAMLIT UI ----------------

st.title("Deepfake Detection System")

mode = st.selectbox(
    "Detection Type",
    ["Image","Video"]
)

model_choice = st.selectbox(
    "Model",
    ["EfficientNet","XceptionNet"]
)

st.divider()


# ---------------- IMAGE DETECTION ----------------

if mode == "Image":

    file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if file:

        original = Image.open(file).convert("RGB")

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(original,width=450)

        if st.button("Detect"):

            frame = cv2.cvtColor(np.array(original),cv2.COLOR_RGB2BGR)

            face = detect_face(frame)

            if face is None:
                st.warning("No face detected")
                st.stop()

            face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(face_rgb)

            if model_choice == "EfficientNet":
                fake,real = predict(efficient_model,face_img)
            else:
                fake,real = predict(xception_model,face_img)

            with col2:
                st.subheader("Detected Face")
                st.image(face_img,width=250)

            st.divider()

            st.subheader("Prediction")

            c1,c2 = st.columns(2)

            c1.metric("Fake Probability",round(fake,3))
            c2.metric("Real Probability",round(real,3))

            if fake > real:
                st.error("Fake Image")
            else:
                st.success("Real Image")


# ---------------- VIDEO DETECTION ----------------

if mode == "Video":

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
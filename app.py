import streamlit as st
st.set_page_config(page_title="Pothole Detection System", layout="wide")

import json
import os
import hashlib
import pandas as pd
import torch
import numpy as np
import tempfile
import cv2
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
USER_DB = "users.json"
HISTORY_DB = "history.json"
COST_PER_CM3 = 50  # INR per cubic cm
COST_PER_px = 0.01  # INR per pixel³
CAMERA_METADATA = {
    "iPhone 14 Pro Max": {"sensor_width_mm": 7.6, "focal_length_mm": 6.86, "native_width_px": 4032},
    "Samsung S23 Ultra": {"sensor_width_mm": 8.4, "focal_length_mm": 6.6, "native_width_px": 4000},
    "Google Pixel 7": {"sensor_width_mm": 7.0, "focal_length_mm": 6.3, "native_width_px": 4000},
}

# --- Auth Utilities ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DB):
        with open(USER_DB, 'w') as f:
            json.dump({}, f)
    with open(USER_DB, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, 'w') as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    hashed = hash_password(password)
    return username in users and users[username] == hashed

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

# --- History Utilities ---
def store_history(username, record):
    record.pop("Image", None)  # Remove image to reduce size
    if not os.path.exists(HISTORY_DB):
        with open(HISTORY_DB, 'w') as f:
            json.dump({}, f)

    with open(HISTORY_DB, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    # Keep only the most recent 49 records before appending the new one
    user_history = data.get(username, [])[-49:]
    user_history.append(record)
    data[username] = user_history

    with open(HISTORY_DB, 'w') as f:
        json.dump(data, f)


def load_history(username):
    if not os.path.exists(HISTORY_DB):
        return []
    with open(HISTORY_DB, 'r') as f:
        data = json.load(f)
    return data.get(username, [])

# --- Model Loading ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(r"E:\\potholeDetector\\runs\\segment\\train8\\weights\\best.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return yolo_model, midas, transform

# --- Area Calculation ---
def calculate_real_world_area(area_px, image_width_px, sensor_width_mm, native_width_px):
    pixel_size_mm = sensor_width_mm / image_width_px
    pixel_area_mm2 = pixel_size_mm ** 2
    area_cm2 = (area_px * pixel_area_mm2) / 10
    if image_width_px < native_width_px:
        scale_factor = (native_width_px / image_width_px) ** 2
        area_cm2 *= scale_factor
    return area_cm2

# --- Login Page ---
def login_page():
    st.title("\U0001F512 Pothole Detection Login")
    mode = st.sidebar.radio("", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Login":
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.page = "main"
                st.rerun()
            else:
                st.error("\u274c Invalid username or password")
    else:
        if st.button("Register"):
            if register_user(username, password):
                st.success("\u2705 Registration successful. You can now login.")
            else:
                st.error("\u26a0\ufe0f Username already exists.")

# --- History Page ---
def history_page():
    st.title("\U0001F5C2 Detection History")
    st.sidebar.button("\U0001F519 Back to Main", on_click=lambda: st.session_state.update({"page": "main"}))
    st.sidebar.button("\U0001F6AA Logout", on_click=lambda: st.session_state.clear())

    records = load_history(st.session_state.username)[-20:]  # limit
    if not records:
        st.info("No history found.")
        return

    for record in records:
        with st.expander(f"\U0001F552 Pothole ID: {record['Pothole ID']}"):
            st.markdown(f"**Timestamp:** {record.get('Timestamp', 'N/A')}")
            st.markdown(f"**Area:** {record['Area']}")
            st.markdown(f"**Depth:** {record['Depth']}")
            st.markdown(f"**Volume:** {record['Volume']}")
            st.markdown(f"**Severity:** {record['Severity']}")
            st.markdown(f"**Repair Cost:** {record['Repair Cost']}")

def process_video_for_potholes(video_path, model, midas_model, midas_transform, metadata=None):
    pothole_frames = []
    summary_records = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Video Duration: {total_frames / fps:.2f} sec | FPS: {fps:.2f} | Total Frames: {total_frames}")

    frame_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model.to(device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 20 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame, conf=0.3, iou=0.4, save=False, verbose=False)

            masks = results[0].masks.data if results[0].masks else []
            img_result = results[0].plot()

            input_tensor = midas_transform(img_rgb).to(device)
            with torch.no_grad():
                depth_prediction = midas_model(input_tensor)
            depth_map = depth_prediction.squeeze().cpu().numpy()
            depth_map_resized = cv2.resize(depth_map, (img_rgb.shape[1], img_rgb.shape[0]))

            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                mask_bin = (mask_np > 0.5).astype(np.uint8)
                indices = np.argwhere(mask_bin)
                if indices.size == 0:
                    continue

                area_px = int(np.sum(mask_bin))
                y_idx, x_idx = indices[:, 0], indices[:, 1]
                h, w = depth_map_resized.shape
                y_idx = np.clip(y_idx, 0, h - 1)
                x_idx = np.clip(x_idx, 0, w - 1)

                avg_depth = np.mean(depth_map_resized[y_idx, x_idx])
                severity = "Severe 🔴" if area_px > 3000 or avg_depth > 0.7 else "Moderate 🟠" if area_px > 1000 else "Small 🟢"
                volume_px3 = area_px * avg_depth
                repair_cost_px = round(volume_px3 * COST_PER_px, 2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pothole_id = f"Frame{frame_count}_Pothole{i+1}"

                if metadata:
                    sensor_width_mm = metadata['sensor_width_mm']
                    native_width_px = metadata['native_width_px']
                    area_cm2 = calculate_real_world_area(area_px, w, sensor_width_mm, native_width_px)
                    depth_cm = avg_depth / 10
                    volume_cm3 = area_cm2 * depth_cm
                    repair_cost = round(volume_cm3 * COST_PER_CM3, 2)
                    record = {
                        "Pothole ID": pothole_id,
                        "Timestamp": timestamp,
                        "Area": f"{area_cm2:.2f} cm²",
                        "Depth": f"{depth_cm:.2f} cm",
                        "Volume": f"{volume_cm3:.2f} cm³",
                        "Severity": severity,
                        "Repair Cost": f"₹{repair_cost}",
                        "Image": img_result.tolist()
                    }
                else:
                    record = {
                        "Pothole ID": pothole_id,
                        "Timestamp": timestamp,
                        "Area": f"{area_px:.2f} px²",
                        "Depth": f"{avg_depth:.2f}",
                        "Volume": f"{volume_px3:.2f}",
                        "Severity": severity,
                        "Repair Cost": f"₹{repair_cost_px}",
                        "Image": img_result.tolist()
                    }

                store_history(st.session_state.username, record)
                summary_records.append(list(record.values()))
            pothole_frames.append(Image.fromarray(img_result))

        frame_count += 1

    cap.release()
    return pothole_frames, summary_records



if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.authenticated:
    login_page()

elif st.session_state.page == "history":
    history_page()

elif st.session_state.page == "main":
    st.sidebar.button("📂 View History", on_click=lambda: st.session_state.update({"page": "history"}))
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.clear())

    st.title("🚗 Pothole Detection & Analysis")

    mode = st.radio("Choose Input Type", ["Upload Image", "Upload Video"])

    conf_slider = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, step=0.05)
    iou_slider = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.4, step=0.05)

    model, midas_model, midas_transform = load_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model.to(device)

    metadata = None
    mode_selected = st.radio("Estimation Mode", ["Relative Estimate Only", "Use Device Metadata"])
    if mode_selected == "Use Device Metadata":
        device_selected = st.selectbox("Select Device", list(CAMERA_METADATA.keys()) + ["Custom"])
        if device_selected != "Custom":
            metadata = CAMERA_METADATA[device_selected]
        else:
            sensor_width = st.number_input("Sensor Width (mm)", min_value=0.0, step=0.1)
            focal_length = st.number_input("Focal Length (mm)", min_value=0.0, step=0.1)
            native_width = st.number_input("Native Image Width (px)", min_value=0, step=100)

            if all(v > 0 for v in [sensor_width, focal_length, native_width]):
                metadata = {
                    "sensor_width_mm": sensor_width,
                    "focal_length_mm": focal_length,
                    "native_width_px": native_width
            }
            else:
                st.warning("Please enter valid non-zero sensor metadata values.")

    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"] if mode == "Upload Image" else ["mp4", "avi", "mov"])

    if st.button("🚀 Run Detection") and uploaded_file:
        if mode == "Upload Image":
            # Handle image
            from PIL import Image
            import cv2
            import tempfile
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            # Detection & display
            results = model.predict(image_np, conf=conf_slider, iou=iou_slider, save=False, verbose=False)

            masks = results[0].masks.data if results[0].masks else []
            img_array = np.array(image)
            detected_img = results[0].plot()
            
            input_tensor = midas_transform(img_array).to(device)
            with torch.no_grad():
                depth_prediction = midas_model(input_tensor)
            depth_map = depth_prediction.squeeze().cpu().numpy()
            depth_map_resized = cv2.resize(depth_map, (img_array.shape[1], img_array.shape[0]))

            st.image([image, Image.fromarray(detected_img)], caption=["Original", "Detected"], use_container_width=True)
            summary = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                mask_bin = (mask_np > 0.5).astype(np.uint8)
                indices = np.argwhere(mask_bin)
                if indices.size == 0:
                    continue
                area_px = int(np.sum(mask_bin))
                y_idx, x_idx = indices[:, 0], indices[:, 1]
                h, w = depth_map_resized.shape
                y_idx = np.clip(y_idx, 0, h - 1)
                x_idx = np.clip(x_idx, 0, w - 1)
                avg_depth = np.mean(depth_map_resized[y_idx, x_idx])
                severity = "Severe 🔴" if area_px > 3000 or avg_depth > 0.7 else "Moderate 🟠" if area_px > 1000 else "Small 🟢"
                volume_px3 = area_px * avg_depth
                repair_cost_px = round(volume_px3 * COST_PER_px, 2)
                if metadata:
                    area_cm2 = calculate_real_world_area(area_px, w, metadata['sensor_width_mm'], metadata['native_width_px'])
                    depth_cm = avg_depth / 10
                    volume_cm3 = area_cm2 * depth_cm
                    repair_cost = round(volume_cm3 * COST_PER_CM3, 2)
                    record = {
                        "Pothole ID": f"Image_{i+1}",
                        "Timestamp": timestamp,
                        "Area": f"{area_cm2:.2f} cm²",
                        "Depth": f"{depth_cm:.2f} cm",
                        "Volume": f"{volume_cm3:.2f} cm³",
                        "Severity": severity,
                        "Repair Cost": f"₹{repair_cost}"
                    }
                else:
                    record = {
                        "Pothole ID": f"Image_{i+1}",
                        "Timestamp": timestamp,
                        "Area": f"{area_px:.2f} px²",
                        "Depth": f"{avg_depth:.2f}",
                        "Volume": f"{volume_px3:.2f}",
                        "Severity": severity,
                        "Repair Cost": f"₹{repair_cost_px}"
                    }
                store_history(st.session_state.username, record)
                summary.append(record)
            st.subheader("📊 Detection Summary")
            df = pd.DataFrame(summary)
            st.dataframe(df)
        else:
            # Handle video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name
            images, summary = process_video_for_potholes(video_path, model, midas_model, midas_transform, metadata)
            st.success(f"Detected potholes in {len(images)} frames.")
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 3].image(img, caption=f"Frame {i+1}", use_container_width=True)
            df = pd.DataFrame(summary, columns=["Pothole ID", "Timestamp", "Area", "Depth", "Volume", "Severity", "Repair Cost"])
            st.subheader("📊 Detection Summary")
            st.dataframe(df)
            os.remove(video_path)

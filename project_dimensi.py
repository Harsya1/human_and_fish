import os
import shutil
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import cv2
import time

# =====================================================
# ============ 1. GUNAKAN FOTO AUGMENTASI ANDA ========
# =====================================================

aug_folder = "augmented"   # Folder berisi SEMUA foto augmentasi kamu

# =====================================================
# ============ 2. BUAT STRUKTUR DATASET ===============
# =====================================================

dataset_path = "dataset"
images_train = f"{dataset_path}/images/train"
images_val = f"{dataset_path}/images/val"
labels_train = f"{dataset_path}/labels/train"
labels_val = f"{dataset_path}/labels/val"

folders = [images_train, images_val, labels_train, labels_val]
for f in folders:
    os.makedirs(f, exist_ok=True)

print("\n=== MEMBANGUN DATASET YOLO DARI FOTO ANDA ===")

all_images = [f for f in os.listdir(aug_folder) if f.lower().endswith((".jpg",".png",".jpeg"))]

train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Copy + buat label dummy YOLO
for f in train_files:
    shutil.copy(f"{aug_folder}/{f}", images_train)
    with open(f"{labels_train}/{os.path.splitext(f)[0]}.txt", "w") as lbl:
        # Dummy bounding box (wajib ada untuk train)
        lbl.write("0 0.5 0.5 1.0 1.0")

for f in val_files:
    shutil.copy(f"{aug_folder}/{f}", images_val)
    with open(f"{labels_val}/{os.path.splitext(f)[0]}.txt", "w") as lbl:
        lbl.write("0 0.5 0.5 1.0 1.0")

print("✔ Dataset train & val selesai dibuat!")

# =====================================================
# ============ 3. BUAT FILE dataset.yaml ==============
# =====================================================

with open(f"{dataset_path}/dataset.yaml", "w") as f:
    f.write(
        f"""
train: {os.path.abspath(images_train)}
val: {os.path.abspath(images_val)}

nc: 1
names: ["fish"]
"""
    )

print("✔ dataset.yaml dibuat!")

# =====================================================
# ================= 4. TRAIN YOLO =====================
# =====================================================

print("\n=== TRAINING YOLOv8 ===")

model = YOLO("yolov8n.yaml")   # model kosong

train_result = model.train(
    data=f"{dataset_path}/dataset.yaml",
    imgsz=640,
    epochs=20,
    batch=16
)

best_model = "runs/detect/train/weights/best.pt"
print("✔ Training selesai! Model terbaik:", best_model)

# =====================================================
# =========== 5. AKURASI COCO VS CUSTOM ===============
# =====================================================

print("\n=== MENGHITUNG AKURASI COCO MODEL ===")
model_coco = YOLO("yolov8n.pt")
res_coco = model_coco.val()

print("\n=== MENGHITUNG AKURASI MODEL ANDA ===")
model_custom = YOLO(best_model)
res_custom = model_custom.val(data=f"{dataset_path}/dataset.yaml")

print("\n=== HASIL AKURASI ===")
print("Akurasi COCO (mAP50):", res_coco.box.map50)
print("Akurasi CUSTOM (mAP50):", res_custom.box.map50)

# =====================================================
# ===== 6. MODE DETEKSI DIMENSI OBJEK (SESUAI PPT) ====
# =====================================================

print("\n=== MODE PENGUKURAN DIMENSI (Tekan ESC untuk keluar) ===")

CARD_WIDTH_CM = 8.56  # lebar kartu ATM/KTP
model_measure = YOLO(best_model)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model_measure(frame)

    for r in results:
        for b in r.boxes:
            conf = float(b.conf)
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            width_pixel = x2 - x1

            # Menghitung skala cm/pixel
            cm_per_pixel = CARD_WIDTH_CM / width_pixel
            object_width = width_pixel * cm_per_pixel

            # Tampilkan ukuran
            cv2.putText(frame, f"{object_width:.2f} cm", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Pengukuran Dimensi Objek", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print("\n=== PROGRAM SELESAI ===")

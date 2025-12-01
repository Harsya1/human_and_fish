import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# --- KONFIGURASI ---
# Menggunakan model hasil training dari project_dimensi.py
MODEL_NAME = 'runs/detect/train/weights/best.pt' 
CSV_FILE = 'output_deteksi_ikan.csv'
CONFIDENCE_THRESHOLD = 0.5

# Inisialisasi Model Custom
if not os.path.exists(MODEL_NAME):
    print(f'Error: Model {MODEL_NAME} tidak ditemukan.')
    print('Jalankan project_dimensi.py terlebih dahulu untuk training model.')
    exit()

try:
    model = YOLO(MODEL_NAME)
    print(f"✔ Model berhasil dimuat: {MODEL_NAME}")
except Exception as e:
    print(f'Error loading model: {e}')
    exit()

# Siapkan File CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Object_Label', 'Confidence'])

cap = cv2.VideoCapture(0) # Atau ganti dengan path video file

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Ambil nama label dari model custom (misal: 'fish', 'shrimp')
        label = model.names[cls_id]
        
        if conf > CONFIDENCE_THRESHOLD:
            # Visualisasi
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Simpan ke CSV
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                # Parameter: Fish (Label) & Confidence
                writer.writerow([datetime.now(), label, f'{conf:.2f}'])

    cv2.imshow('Deteksi Ikan', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

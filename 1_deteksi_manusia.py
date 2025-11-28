import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# --- KONFIGURASI ---
MODEL_NAME = 'yolov8n.pt'  # Model bawaan (COCO)
TARGET_CLASS_ID = 0        # ID 0 adalah 'person' di COCO
CSV_FILE = 'output_deteksi_manusia.csv'
CONFIDENCE_THRESHOLD = 0.5

# Inisialisasi Model
model = YOLO(MODEL_NAME)

# Siapkan File CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Object_Label', 'Confidence'])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Filter hanya Manusia (Person)
        if cls_id == TARGET_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
            label = model.names[cls_id]
            
            # Visualisasi
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Simpan ke CSV
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                # Parameter: Human (Label) & Confidence
                writer.writerow([datetime.now(), label, f'{conf:.2f}'])

    cv2.imshow('Deteksi Manusia', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

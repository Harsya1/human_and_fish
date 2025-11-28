# YOLO Object Detection & Analysis

A comprehensive project for real-time object detection using YOLOv8, featuring specialized models for human detection and custom fish/shrimp detection with automated CSV logging and confusion matrix analysis.

## Project Overview

This project implements three main components:
1. Human detection using pre-trained COCO model
2. Fish/shrimp detection using custom-trained model
3. Confusion matrix generation for model evaluation

## Project Structure

```
human_and_fish/
├── 1_deteksi_manusia.py              # Human detection script
├── 2_deteksi_ikan.py                 # Fish detection script
├── 3_confusion_matrix_generator.py   # Confusion matrix visualization
├── output_deteksi_manusia.csv        # Human detection results
├── yolov8n.pt                        # YOLOv8 nano model weights
└── README.md                         # This file
```

## Requirements

- Python 3.7+
- opencv-python
- ultralytics
- matplotlib
- seaborn
- scikit-learn
- numpy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Harsya1/human_and_fish.git
cd human_and_fish
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python ultralytics matplotlib seaborn scikit-learn numpy
```

## Usage

### Human Detection (1_deteksi_manusia.py)

Detects humans in real-time video stream using the pre-trained YOLOv8 nano model.

```bash
python 1_deteksi_manusia.py
```

Features:
- Real-time human detection from webcam
- Automatic CSV logging with timestamp and confidence score
- Bounding box visualization
- Confidence threshold filtering (default: 0.5)

**Output:** `output_deteksi_manusia.csv` containing:
- Timestamp
- Object Label (person)
- Confidence score

Press 'q' to quit the application.

### Fish Detection (2_deteksi_ikan.py)

Detects fish and shrimp objects using a custom-trained YOLO model.

```bash
python 2_deteksi_ikan.py
```

Features:
- Custom model support for specialized object detection
- Real-time detection from webcam or video file
- CSV logging of all detections
- Error handling for missing model files
- Bounding box visualization

**Output:** `output_deteksi_ikan.csv` containing:
- Timestamp
- Object Label (fish/shrimp type)
- Confidence score

**Note:** Requires `best_fish.pt` model file trained on your custom dataset. Update `MODEL_NAME` configuration if using a different model path.

Press 'q' to quit the application.

### Confusion Matrix Generator (3_confusion_matrix_generator.py)

Generates confusion matrix visualization to evaluate model performance.

```bash
python 3_confusion_matrix_generator.py
```

This script:
- Compares predicted results with ground truth data
- Generates heatmap visualization
- Displays True Positives, True Negatives, False Positives, False Negatives

**Output:** Visual confusion matrix plot showing model accuracy metrics.

## Configuration

### 1_deteksi_manusia.py

```python
MODEL_NAME = 'yolov8n.pt'        # YOLO model file
TARGET_CLASS_ID = 0               # Class ID for 'person'
CSV_FILE = 'output_deteksi_manusia.csv'
CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence score
```

### 2_deteksi_ikan.py

```python
MODEL_NAME = 'best_fish.pt'       # Custom trained model
CSV_FILE = 'output_deteksi_ikan.csv'
CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence score
```

## Model Information

### Human Detection
- **Model:** YOLOv8 Nano (COCO)
- **Classes:** 80 standard COCO classes
- **Key Class:** Person (ID: 0)
- **Advantage:** Pre-trained, no additional training required

### Fish Detection
- **Model:** Custom-trained YOLOv8
- **Training:** Requires manual dataset annotation and model training
- **Customizable:** Adapt labels and classes to your specific use case

## Workflow

1. **Data Acquisition:** Captures video frames from webcam in real-time
2. **Inference:** Processes frames through YOLO model
3. **Filtering:** Applies confidence threshold and class filtering
4. **Logging:** Records detections to CSV for analysis
5. **Visualization:** Displays bounding boxes on frames

## Output CSV Format

Both detection scripts produce CSV files with the following columns:

| Timestamp | Object_Label | Confidence |
|-----------|--------------|-----------|
| 2025-11-28 10:30:45.123456 | person | 0.92 |
| 2025-11-28 10:30:45.234567 | person | 0.87 |

## Confusion Matrix Metrics

The confusion matrix displays four key metrics:

- **True Positive (TP):** Object present and correctly detected
- **True Negative (TN):** No object and correctly not detected
- **False Positive (FP):** No object but incorrectly detected (false alarm)
- **False Negative (FN):** Object present but not detected (missed detection)

## Troubleshooting

### Model file not found
Ensure the model file exists in the working directory or provide the full path in the configuration.

### Webcam not detected
- Check if another application is using the webcam
- Try specifying a video file path instead: `cv2.VideoCapture('video.mp4')`

### CSV file permission error
Ensure the directory has write permissions and close any applications that may be locking the CSV file.

### CUDA/GPU issues
The ultralytics library will automatically fall back to CPU if CUDA is not available.

## Advanced Usage

### Using video files instead of webcam
Replace the VideoCapture line:
```python
# From webcam
cap = cv2.VideoCapture(0)

# From video file
cap = cv2.VideoCapture('path/to/video.mp4')
```

### Adjusting confidence threshold
Modify the `CONFIDENCE_THRESHOLD` variable:
```python
CONFIDENCE_THRESHOLD = 0.7  # Higher = fewer false positives
```

### Custom model training
To train a custom fish detection model:
1. Collect and annotate dataset
2. Use YOLO format labels (YOLO format or convert from COCO)
3. Train using ultralytics: `model.train(data='dataset.yaml', epochs=100)`

## Performance Considerations

- YOLOv8 Nano: Lightweight, suitable for real-time inference on CPU
- For faster GPU inference, consider using GPU-enabled CUDA installation
- Frame processing rate depends on model size and hardware

## Future Enhancements

- Multi-object tracking across frames
- Batch processing for video files
- Database integration for large-scale logging
- Web dashboard for real-time monitoring
- Model performance metrics dashboard

## References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- COCO Dataset: https://cocodataset.org/
- scikit-learn Metrics: https://scikit-learn.org/stable/modules/metrics.html

## License

This project is provided as-is for educational and research purposes.

## Author

Harsya1

## Support

For issues and questions, please refer to the project repository or contact the maintainer.

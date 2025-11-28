import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- DATA SIMULASI (CONTOH) ---
# Dalam kasus nyata, 'y_actual' berasal dari anotasi manual dataset uji Anda
# 'y_pred' berasal dari hasil deteksi model (file CSV output)

# 1 = Objek Terdeteksi (e.g., Ikan), 0 = Background/Tidak ada
y_actual = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0] # Kunci Jawaban
y_pred   = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] # Hasil Model

# Membuat Confusion Matrix
cm = confusion_matrix(y_actual, y_pred)

# Plotting
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Deteksi Ikan')
plt.show()

# Penjelasan Output:
# TP (True Positive): Ikan ada, dan terdeteksi ikan.
# TN (True Negative): Tidak ada ikan, dan tidak terdeteksi apa-apa.
# FP (False Positive): Tidak ada ikan, tapi model bilang ada (Salah Deteksi).
# FN (False Negative): Ikan ada, tapi model tidak melihatnya (Meleset).

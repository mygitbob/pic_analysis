import cv2
import numpy as np
import tensorflow as tf

# Lade das vortrainierte Modell von TensorFlow
model = tf.keras.models.load_model('ssd_inception_v2.pb')

# Lade das Bild
img = cv2.imread('immobilien_bild.jpg')

# Konvertiere das Bild in ein 4D-Tensor und führe Histogramm-Ausgleich durch
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.equalizeHist(img)

# Wende Median-Filterung an, um das Rauschen zu reduzieren
img = cv2.medianBlur(img, 3)

# Konvertiere das Bild zurück in das BGR-Format
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Vorhersage mit dem Modell
img = cv2.resize(img, (300, 300))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
preds = model.predict(img)

# Extrahiere die Bounding Box-Koordinaten des Briefumschlags aus der Vorhersage
boxes = preds[0, :, :4]
scores = preds[0, :, 4:]
best_box = np.argmax(scores)
box = boxes[best_box]

# Extrahiere die Koordinaten des Briefumschlags aus der Bounding Box
x1 = int(box[1] * img


import cv2
import numpy as np
import tensorflow as tf

# Lade das vortrainierte Modell von TensorFlow
model = tf.keras.models.load_model('ssd_inception_v2.pb')

# Lade das Bild
img = cv2.imread('immobilien_bild.jpg')

# Farbkorrektur mit dem Automatischen Weißabgleich
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)
img = cv2.merge((l,a,b))
img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

# Konvertiere das Bild in ein 4D-Tensor und führe Histogramm-Ausgleich durch
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.equalizeHist(img)

# Wende Median-Filterung an, um das Rauschen zu reduzieren
img = cv2.medianBlur(img, 3)

# Wende Schwellenwertbildung an, um den Briefumschlag vom Hintergrund zu trennen
_, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# Konvertiere das Bild zurück in das BGR-Format
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
x1 = int(box[1] * img.shape[2])
y1 = int(box[0] * img.shape[1])
x2 = int(box[3] * img.shape[2])
y2 = int(box[2] * img.shape[1])

# Zeichne ein Rechteck um den Briefumschlag
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Speichere das Ergebnisbild
cv2.imwrite('briefumschlag.jpg', img)


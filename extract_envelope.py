import cv2
import numpy as np
import tensorflow as tf

# Lade das vortrainierte Modell von TensorFlow
model = tf.keras.models.load_model('ssd_inception_v2.pb')

# Lade das Bild
img = cv2.imread('immobilien_bild.jpg')

# Konvertiere das Bild in ein 4D-Tensor
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)

# Vorhersage mit dem Modell
preds = model.predict(img)

# Extrahiere die Bounding Box-Koordinaten des Briefumschlags aus der Vorhersage
boxes = preds[0, :, :4]
scores = preds[0, :, 4:]

# Wähle die beste Bounding Box aus, die den Briefumschlag enthält
best_box = np.argmax(scores)
box = boxes[best_box]

# Extrahiere die Koordinaten des Briefumschlags aus der Bounding Box
x1 = int(box[1] * img.shape[2])
y1 = int(box[0] * img.shape[1])
x2 = int(box[3] * img.shape[2])
y2 = int(box[2] * img.shape[1])

# Zeichne die Bounding Box um den Briefumschlag
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Speichere das Bild mit der Bounding Box
cv2.imwrite('immobilien_bild_mit_box.jpg', img)


from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.keras')

img_path = './image.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image

# Predict
prediction = model.predict(img_array)
print(prediction)

if prediction[0] > 0.6:
    print("This image contains sunglasses.")
else:
    print("This image does not contain sunglasses.")

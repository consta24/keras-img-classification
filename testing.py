from tensorflow import keras
from tensorflow.keras.utils import load_img
import numpy as np
 
model = keras.models.load_model('model_saved.h5')

testPlanes = np.array

for i in range(1, 101):
    if i <= 50:
        img_path = 'v_data/test/cars/' + str(i) + '.jpg'
    else:
        img_path = 'v_data/test/planes/' + str(i-50) + '.jpg'
    image = load_img(img_path, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,224,224,3)
    label = model.predict(img)
    score = label[0]
    print(
        img_path + " is %.2f percent car and %.2f percent plane."
        % (100 * (1 - score), 100 * score)
    )

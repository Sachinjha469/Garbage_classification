# Part 3 - Making new predictions
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model = load_model('model.h5')
test_image = image.load_img('Dataset/predict/identify.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Garbage'
else:
    prediction = 'Not_Garbage'

print(prediction) 

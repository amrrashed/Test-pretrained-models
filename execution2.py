#test pretrained mode
import time
start = time.time()

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50

# load the model
model = ResNet50()
# load an image from file
#image = load_img('mug.jpg', target_size=(224, 224))#coffee_mug (95.70%)
image = load_img('cat.jpg', target_size=(224, 224))#Egyptian_cat (95.75%)
#image = load_img('dog.jpg', target_size=(224, 224))#German_shepherd (92.94%)
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
end = time.time()
print("Execution time: {0:.5} seconds \n".format(end-start))


from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import imutils
# load the image
image = cv2.imread('2.jpg')
output = imutils.resize(image, width=400)
# classes = ['black', 'blue', 'brown', 'green', 'grey', 'metallic', 'multicolour', 'nude & neutrals', 'pink & purple', 'red', 'white', 'yellow & orange']
# {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'grey': 4, 'metallic': 5, 'multicolour': 6, 'nd': 7, 'nude & neutrals': 8, 'pink & purple': 9, 'red': 10, 'white': 11, 'yellow & orange': 12}

classes = ['black', 'blue', 'brown', 'green', 'grey', 'metallic', 'multicolour', 'nude&neutrals', 'pink & purple', 'red', 'white', 'yellow & orange']
# pre-process the image for classification
image = cv2.resize(image, (128, 128))
image = image.astype("float")//255
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
print("[INFO] loading network...")
model = load_model('./weights/weights.h5')

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(classes[j], proba[j] * 100)
    cv2.putText(output, label, (10, (i * 30) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(classes, proba):
    print("{}: {:.2f}%".format(label, p * 100))

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
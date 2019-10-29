import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet_do_bn import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb =LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", 'ship', "truck"]
NUM_EPOCHS = 40
INIT_LR = 1e-2
# initialize optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR/NUM_EPOCHS, momentum=0.9, nesterov=True)
#opt = SGD(lr=INIT_LR, decay=0, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the network 
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)

#evaluate the network 
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot training and loss accuracy

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,NUM_EPOCHS), H.history["loss"],label="train_loss")
plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_loss"],label="val_loss")
#print("history keys: %s" % list(H.history.keys()))
plt.plot(np.arange(0,NUM_EPOCHS), H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_accuracy"],label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10 (with LR decay)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])




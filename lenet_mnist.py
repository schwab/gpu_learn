from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from nn.conv.lenet import LeNet
from sklearn.datasets import fetch_openml
import os
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="path to saved model")
args = vars(ap.parse_args())

print("[INFO] accessing MNIST...")
    #dataset = datasets.fetch_mldata("MNIST Original")
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')

dataset = mnist
data = mnist.data

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28,28,1)

(trainX, testX, trainY, testY) = train_test_split(data/ 255.0,
    dataset.target.astype("int"), test_size=0.25,random_state=42)

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

if "model" in args and os.path.exists(args["model"]):
    model = load_model(args["model"])
else:
    # initialize the optimizer and model
    print("[INFO] compiling model ...")
    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), 
        batch_size=128, epochs=20, verbose=1)
    if "model" in args:
        model.save(args["model"])
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,20), H.history["loss"],label="train_loss")
    plt.plot(np.arange(0,20), H.history["val_loss"],label="val_loss")
    #print("history keys: %s" % list(H.history.keys()))
    plt.plot(np.arange(0,20), H.history["accuracy"],label="train_acc")
    plt.plot(np.arange(0,20), H.history["val_accuracy"],label="val_acc")
    plt.xlabel("Training Loss and Accuracy")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

# evaluate the network
print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), 
    predictions.argmax(axis=1), 
    target_names=[str(x) for x in le.classes_]))



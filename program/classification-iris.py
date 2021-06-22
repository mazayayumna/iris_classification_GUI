# imports and preliminaries
import csv
import numpy as np
import keras as kr
import tensorflow as tf

# Load the Iris dataset.
iris = list(csv.reader(open('iris.csv')))[1:]

# The inputs are four floats: sepal length, sepal width, petal length, petal width.
inputs  = np.array(iris)[:,:4].astype(np.float)

# Outputs are initially individual strings: setosa, versicolor or virginica.
outputs = np.array(iris)[:,4]

# Convert the output strings to ints.
outputs_vals, outputs_ints = np.unique(outputs, return_inverse=True)

# Encode the category integers as binary categorical vairables.
outputs_cats = tf.keras.utils.to_categorical(outputs_ints)

# Split the input and output data sets into training and test subsets.
inds = np.random.permutation(len(inputs))
train_inds, test_inds = np.array_split(inds, 2)
inputs_train, outputs_train = inputs[train_inds], outputs_cats[train_inds]
inputs_test,  outputs_test  = inputs[test_inds],  outputs_cats[test_inds]

# Create a neural network.
model = kr.models.Sequential()

# Add an initial layer with 4 input nodes, and a hidden layer with 16 nodes.
model.add(kr.layers.Dense(16, input_shape=(4,)))
# Apply the sigmoid activation function to that layer.
model.add(kr.layers.Activation("sigmoid"))
# Add another layer, connected to the layer with 16 nodes, containing three output nodes.
model.add(kr.layers.Dense(3))
# Use the softmax activation function there.
model.add(kr.layers.Activation("softmax"))
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the model using our training data.
model.fit(inputs_train, outputs_train, epochs=50, batch_size=1, verbose=1)

# Evaluate the model using the test data set.
loss, accuracy = model.evaluate(inputs_test, outputs_test, verbose=1)

# Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
# import Ui_iriseus
from Ui_iriseus import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # button
        self.ui.pushButton.setText('Prediksi')
        self.ui.pushButton.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        sl = float(self.ui.lineEdit.text())
        sw = float(self.ui.lineEdit_2.text())
        pl = float(self.ui.lineEdit_3.text())
        pw = float(self.ui.lineEdit_4.text()) #ubah ke float zheyeng ini string
        testVal = [[sl, sw, pl, pw]]
        prediction = model.predict(testVal)
        pIndex = np.argmax(prediction, axis=1)[0]
        classNames = ["Setosa", "Versicolor", "Virginica"]
        hasil = classNames[pIndex]
        self.ui.label.setText(hasil)
 
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

'''if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_iriseus.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())'''
 

# virginica: [5.8,2.7,5.1,1.9]
# versicolor: [6.4, 3.2, 4.5, 1.5]
# setosa: [4.8, 3.0, 1.4, 0.1]
'''sl = 6.4
sw = 3.2
pl = 4.5
pw = 1.5
testVal = [[sl, sw, pl, pw]]
prediction = model.predict(testVal)
pIndex = np.argmax(prediction, axis=1)[0]
classNames = ["Setosa", "Versicolor", "Virginica"]
print(classNames[pIndex])'''

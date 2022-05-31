import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

tic = time.process_time() #save start time 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #get data 
x_train, x_test = x_train / 255.0, x_test / 255.0 #normalize input data between 0 and 1

model = Sequential() 
model.add(Flatten(input_shape=(28, 28))) #28 x 28 image matrix -> 784 element vector 
model.add(Dense(64)) #128 neurons with logistic activation function
#model.add(Dense(10, activation = "sigmoid")) #10 neurons with logistic activation function

loss_fn = SparseCategoricalCrossentropy(from_logits=True) #define loss function 
#since there are 10 outputs, crossentropy needs to be used as loss 
#the outputs are not normalized so from_logits needs to be true

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1) #train
model.evaluate(x_test, y_test, verbose=2) #test

weights = model.get_weights()
print("\nLayer Weight Counts")
for i in range(0, len(model.layers) * 2 - 2, 2):
    print("Layer" + str(int(i / 2)) + ": ", end = "")
    print("weights(" + str(weights[i].shape[0] * weights[i].shape[1]) + "),", "biases(" + str(weights[i + 1].shape[0]) + ")")

with open('weights.txt', 'w') as f:
    for i in range(0, len(model.layers) * 2 - 2, 2):
        for i in range(0, len(model.layers) * 2 - 2, 2):
            for neuron in weights[i].tolist():
                f.write(" ".join([str(w) for w in neuron]) + "\n")
            f.write(" ".join([str(b) for b in weights[i + 1]]) + "\n\n")
    
toc = time.process_time()
print("\nTime elapsed:", str(toc - tic) + "s")
#Albert Tang P4 2023


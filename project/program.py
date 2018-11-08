import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the training data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Model definitions
def model_1(model):
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


def model_2(model):
    model.add(tf.keras.layers.Dense(12, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


def model_3(model):
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# Test a given model and print the loss and accuracy
def test_model(model_spec, epochs=4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model_spec(model)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs)  # train the model

    val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the model using the test data
    print("Loss: %.4f, accuracy: %.4f" % (val_loss, val_acc))


#print("Model 1:")
#test_model(model_1, epochs=4)

print("Model 3:")
test_model(model_3, epochs=20)

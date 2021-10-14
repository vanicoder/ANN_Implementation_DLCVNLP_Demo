import tensorflow as tf

def get_data(validation_data_size):
    mnist=tf.keras.datasets.mnist
    (X_train_full,Y_train_full),(X_test,Y_test)=mnist.load_data()

    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_data_size] / 255., X_train_full[validation_data_size:] / 255.
    Y_valid, Y_train = Y_train_full[:validation_data_size], Y_train_full[validation_data_size:]

    # scale the test set as well
    X_test = X_test / 255.

    return (X_train,Y_train),(X_valid,Y_valid),(X_test,Y_test)
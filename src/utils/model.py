import tensorflow as tf
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_model(Loss_Function,Optimizer,Metrics,No_of_Classes):
    Layers=[
        tf.keras.layers.Flatten(input_shape=[28,28],name="inputLayer"), # name of the 1st layer is inputlayer that we are assigning here. Naming is not requd it's upto u if u want to give 
        tf.keras.layers.Dense(300,activation="relu",name="hiddenlayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hiddenlayer2"),
        tf.keras.layers.Dense(No_of_Classes,activation="softmax",name="outputlayer")
    ]
    model_clf=tf.keras.Sequential(Layers)
    model_clf.summary()

    # Compiling the model
    model_clf.compile(loss=Loss_Function,optimizer=Optimizer,metrics=Metrics)

    return model_clf

def get_unique_filename(filename):
        unique_filename=time.strftime(f"%Y%m%d_%H%M%S_{filename}")
        return unique_filename

def save_model(model,model_name,model_dir):
        unique_filename=get_unique_filename(model_name)
        path_to_model=os.path.join(model_dir,unique_filename)
        model.save(path_to_model)

def save_plot(history,plot_dir):
    pd.DataFrame(history.history).plot(figsize=(10, 8))
    plt.grid(True)
    os.makedirs(plot_dir,exist_ok=True)
    plot_file = os.path.join(plot_dir, "loss_accuracy.png")
    plt.savefig(plot_file)




from src.utils.common import  read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model,save_plot
import argparse
import os

def training(config_path):
    config=read_config(config_path)
    validation_data_size=config['Params']['validation_datasize']
    (X_train,Y_train),(X_valid,Y_valid),(X_test,Y_test)=get_data(validation_data_size)
    Loss_Function=config['Params']['loss_function']
    Optimizer=config['Params']['optimizer']
    Metrics=config['Params']['metrics']
    No_of_Classes=config['Params']['no_classes']
    model_clf=create_model(Loss_Function,Optimizer,Metrics,No_of_Classes)

    Epochs=config['Params']['epochs']
    Validation_Data=(X_valid,Y_valid)
    history=model_clf.fit(X_train,Y_train,epochs=Epochs,validation_data=Validation_Data)

    model_name=config['artifacts']['model_name']
    artifact_dir=config['artifacts']['artifacts_dir']
    model_dir=config['artifacts']['model_dir']
    model_dir_path=os.path.join(artifact_dir,model_dir)

    save_model(model_clf,model_name,model_dir_path)

    plot_dir = config["artifacts"]["plots_dir"]
    #logger.info("Plot Loss/Accuracy curves..")
    save_plot(history, plot_dir)


if __name__=='__main__':
    args=argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config.yaml",help="Use to add arguments in run time")
    parsed_args=args.parse_args()
    training(config_path=parsed_args.config)

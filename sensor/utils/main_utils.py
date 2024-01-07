import yaml
from sensor.exception import SensorException
from sensor.logger import logging
import numpy as np
import dill
import sys,os

def read_yaml_file(file_path:str) ->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(e,sys)
    

def write_yaml_file(file_path:str, content:object, replace:bool = False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise SensorException(e,sys)
    
def save_numpy_array_data(file_path:str, array:np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise SensorException(e,sys)
    

def load_numpy_array_data(file_path:str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return :np.array data loaded
    """

    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e,sys)
    
def save_object(file_path:str, obj:object) -> None:
    try:
        logging.info('Executing save_object of main_utils.py')
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info('Exiting save_object of main_utils.py')
    except Exception as e:
        raise SensorException(e,sys)
    


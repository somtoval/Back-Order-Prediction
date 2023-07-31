import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #This is just a string format on how the name of the log file will be 
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) #Joining this current working directory to the above string
os.makedirs(logs_path, exist_ok=True) #Creates the logs_path directory above

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) # joining the path of the logs_path to the log_file name string making the directory a folder and storing a the log file there

# print('log file:', LOG_FILE, 'logs_path:', logs_path, 'LOG_FILE_PAth:', LOG_FILE_PATH, os.getcwd())

# Creating a logging instance (I think that this logging module we imported is a class and basicConfig is a class method where by it doesn't need an object, it can access the class resources without creating an instance)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

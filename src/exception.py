import sys
from src.logger import logging # I'm understand the OOP stuff even more, this logging module we are import here is the python logging module which we imported in our logger.py but there we have customized the class method

# The function takes in one argument which is "error_detail" and it will be an object of the python sys which will give us info about error
def error_message_detail(error,error_detail:sys):

    # Here the python "sys" .exc_info() method returns a full traceback detail of the error
    _,_,exc_tb = error_detail.exc_info()
    # We get the file name from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Here we create a comprehensive message
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    # We return a comprehensive error message
    return error_message


# Now we create a Custom Class exception inheriting from python default exception class
class CustomException(Exception):
    
    # Takes input of message and error detail which is an object of sys
    def __init__(self, error_message, error_detail:sys):
        # Passing our error message to python builtin Exception class(I think that the python builtin exception class has a argument that takes in the error)
        super().__init__(error_message)
        # the error message argument passed in was assigned to a variable with the error message and error detail as it's argument
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    # Returning the message
    def __str__(self):
        return self.error_message  
import sys
from network_security.logging import logger

class NetworkSecurityException(Exception):
    """
    Custom exception class for the Network Security project.
    It captures the file name and line number where the exception occurred,
    along with the original error message.
    """
    def __init__(self,error_message,error_details:sys):
        """
        Initializes the exception with error message and traceback details.

        Parameters:
        - error_message: The actual exception raised.
        - error_details: sys module, used to extract the traceback info.
        """
        self.error_message = error_message
        # Get the traceback object from the error details
        _,_,exc_tb = error_details.exc_info()
        
        # Extract the line number and file name where the exception occurred
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
    def __str__(self):
        """
        Custom string representation of the exception.
        """
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
        
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block")
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
           raise NetworkSecurityException(e,sys)
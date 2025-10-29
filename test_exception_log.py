import os 
import sys

from src.logger import get_log
from src.custom_exception import CustomException

logger = get_log(__name__)

def divide_num(a: int, b: int):
    try:
        result = a / b
        logger.info("Dividing two numbers")
        return result
    except Exception as e:
        logger.error("Error zero division")
        raise CustomException("Error in dividing by zero",sys)

if __name__ == "__main__":
    try:
        logger.info("starting main")
        divide_num(10,0)
    except CustomException as ce:
        logger.error(str(ce))
        print(ce)
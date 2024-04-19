import string
import random
from datetime import datetime
from omegaconf import DictConfig

def timestamp_str() -> str:
    """ Get current time stamp string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def random_str(length: int=4) -> str:
    """ Generate random string with given length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))


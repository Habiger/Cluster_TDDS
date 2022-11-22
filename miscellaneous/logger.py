import logging
import datetime

def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("MC_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"logs/MC_run_{now}.log")
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


def start_logger_if_necessary():
    logger = logging.getLogger("MC_logger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"logs/MC_run_{now}.log", mode='a')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


import logging
import datetime
import glob
import os
import shutil


def get_log_filepath():
    cwd = os.getcwd()
    default_path = os.path.join(cwd, "3_mc_results")  # per default logfiles will be stored together with the clustering results
    if not os.path.exists(default_path):
        alternative_path = os.path.join(cwd, "logs")           # for debugging
        if not os.path.exists(alternative_path):
            os.makedirs(alternative_path)
    else: 
        return default_path

class Logger:
    logger_name: str = "Clustering"
    path_logfiles: str = get_log_filepath()     
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def start_logger_if_necessary(cls):
        """Creates/reloads the logger and returns it
        """
        logger = logging.getLogger(f"{cls.logger_name}_logger")
        if len(logger.handlers) == 0: # because of a known joblib bug: https://github.com/joblib/joblib/issues/1017
            logger.setLevel(logging.INFO)
            # create the logging file handler
            file_path = os.path.join(cls.path_logfiles, f"{cls.logger_name}.log")
            fh = logging.FileHandler(file_path, mode="a")
            formatter = logging.Formatter(cls.fmt)
            fh.setFormatter(formatter)
            # add handler to logger object
            logger.addHandler(fh)
        return logger 


    @classmethod
    def close_logger(cls):
        """Removes all filehandlers and calls logging.shutdown(). Will be called at the end of a clustering run.
        """
        logger = logging.getLogger(f"{cls.logger_name}_logger")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logging.shutdown()




""" old methods: for handling joblib bug (didnt work well)

    @classmethod
    def move_logfile(cls, destination_path):
        logfilepath = os.path.join(cls.path_logfiles, f"{cls.logger_name}.log")
        shutil.move(logfilepath, destination_path)

    @classmethod
    def _get_latest_log_filepath(cls):   # currently not used
        ""recursively searches for logfile path with the latest timestamp
        ""
        search_path = os.path.join(cls.path_logfiles, f"{cls.logger_name}.log")
        return sorted(glob.glob(search_path))[-1]


    @classmethod
    def reload_logger(cls):  #currently not used
        ""reloads the last generated logger in case the joblib parallel module as no access to it
        ""
        logger = logging.getLogger(f"{cls.logger_name}_logger")
        if len(logger.handlers) == 0:   
            logger.setLevel(logging.INFO)
            file_path = cls._get_latest_log_filepath()
            fh = logging.FileHandler(file_path, mode='a')
            fh.setFormatter(logging.Formatter(cls.fmt))
            logger.addHandler(fh)
            logger.warning("\nreload_logger\n\n")
        return logger

"""
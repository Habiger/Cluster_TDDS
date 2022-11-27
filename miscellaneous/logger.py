import logging
import datetime
import glob
import os




class Logger:
    logger_prefix = "MC"
    directory_logfiles = "logs"

    @classmethod
    def create_logger(cls):
        """Creates a logging object and returns it
        """
        logger = logging.getLogger(f"{cls.logger_prefix}_logger")
        logger.setLevel(logging.INFO)
        # create the logging file handler
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"logs/{cls.logger_prefix}_run_{now}.log")
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        # add handler to logger object
        logger.addHandler(fh)
        return logger 

    @classmethod
    def _get_latest_log_filepath(cls):
        """searches for logfile path with the latest timestamp
        """
        search_path = os.path.join(cls.directory_logfiles, f"{cls.logger_prefix}*.log")
        return sorted(glob.glob(search_path))[-1]
    
    @classmethod
    def reload_logger(cls):
        """reloads the last generated logger in case the joblib parallel module as no access to it
        """
        logger = logging.getLogger(f"{cls.logger_prefix}_logger")
        if len(logger.handlers) == 0:   # joblib parallel sometimes hasn´t access to filehandlers
            logger.setLevel(logging.INFO)
            path = cls._get_latest_log_filepath()
            fh = logging.FileHandler(path, mode='a')
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
            logger.addHandler(fh)
        return logger

    @classmethod
    def close_logger(cls):
        """removes all filehandlers and calls logging.shutdown()
        """
        logger = logging.getLogger(f"{cls.logger_prefix}_logger")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logging.shutdown()


import logging
import os
import time


class Logger:
    def __init__(self, name="default_logger", log_dir="logs", logging_mode=True):
        """
        Initialize the Logger instance.

        Args:
            name (str): Name of the logger.
            log_dir (str): Directory where the log files will be saved.
            logging_mode (bool): If True, logs will be written to a file, otherwise printed to console.
        """
        self.logger = logging.getLogger(name)
        self.logging_mode = logging_mode
        if logging_mode:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            log_file = f"{log_dir}/{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def log(self, msg):
        """
        Log a message.

        Args:
            msg (str): The message to log.
        """
        if self.logging_mode:
            self.logger.info(msg)
        else:
            print(msg)

    def rename_log(self):
        """
        rename the log file.
        """
        if self.logging_mode:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    path_name = os.path.dirname(handler.baseFilename)
                    file_name = "_eval_" + os.path.basename(handler.baseFilename)
                    os.rename(
                        src=handler.baseFilename, dst=os.path.join(path_name, file_name)
                    )


def eval_log(logger):
    """
    Decorator to log errors and rename logs upon successful execution.

    Args:
        logger (Logger): Logger instance to handle logging and renaming.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logger.rename_log()
                return result
            except Exception as e:
                logger.log(f"Error: {e}")
                raise

        return wrapper

    return decorator

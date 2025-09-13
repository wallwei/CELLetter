import logging

def setup_logger(log_file='lri_prediction.log'):
    """Setup logger configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode='w'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger
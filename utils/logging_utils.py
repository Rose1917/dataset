import logging
def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    # set logging with color
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.DEBUG, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))  



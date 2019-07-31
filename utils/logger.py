# use logger.info(...) instead of print(...)
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', # Change to INFO for production usage
                    fmt='%(asctime)s %(levelname)s %(message)s')

# block useless warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
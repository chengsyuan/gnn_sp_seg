# use logger.info(...) instead of print(...)
from utils.conf import conf
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level=conf.level, # Change to INFO for production usage
                    fmt='%(asctime)s %(levelname)s %(message)s')

# block useless warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
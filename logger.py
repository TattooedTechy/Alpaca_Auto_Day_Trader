#logger.py
import logging
from logging.handlers import RotatingFileHandler
import sys

# Create your central logger
log = logging.getLogger("TradingBot")
log.setLevel(logging.INFO)

# Formatter for all handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# file
fh = RotatingFileHandler(
    "trading_bot.log",      # log filename
    mode="a",              
    maxBytes=10 * 1024 * 1024,  # rotate after 10 MB
    backupCount=5,             # keep up to 5 old files
    encoding="utf-8"
)
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

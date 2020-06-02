# -*- coding: utf-8 -*-

import re
import os
import logging 

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)


### Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"/logs/embed_selectedfolders.log"),
        logging.StreamHandler()
    ])

#LOG = logging.getLogger('LOG')
#LOG.setLevel(logging.DEBUG)
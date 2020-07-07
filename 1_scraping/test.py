print('beginning')

import logging

with open('cwd.txt','r') as file: rootfold = file.read().rstrip()+'/'

print(rootfold)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s %(name)10s [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"logs/test.log"),
        logging.StreamHandler()
    ])

LOG = logging.getLogger('LOG')
LOG.setLevel(logging.DEBUG)

LOG.info('functioneaza')

print('end')

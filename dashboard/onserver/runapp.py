import psutil
import sys
from subprocess import Popen

for process in psutil.process_iter():
    if process.cmdline() == ['python', '/home/ubuntu/dashboard/APP_corona_v03.py']:
        sys.exit('Process found: exiting.')

print('Process not found: starting it.')
Popen(['python', '/home/ubuntu/dashboard/APP_corona_v03.py'])

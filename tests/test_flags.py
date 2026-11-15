# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd

rootfold = re.match('^.*reddit-disinformation', 
                    os.path.dirname(os.path.realpath(__file__))).group(0)
sys.path.append(rootfold+"/2_findflags/all_subreddits")
import functions_pos_match as fpm


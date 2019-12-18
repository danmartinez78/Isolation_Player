import traceback
from robot_player import CustomPlayer as Lucy
from human_player import HumanPlayer
from test_players import HumanPlayer as BasicHumanPlayer
from isolation import Board, game_as_text
import platform
import random
import perception

if platform.system() != 'Windows':
    import resource
from time import time, sleep

import os
import sys
import time
import functools
sys.path.append(os.path.join(os.path.dirname(__file__), './uArm-Python-SDK/'))
from uarm.wrapper import SwiftAPI

# prompt difficulty

# instantiate robot player

# instantiate human player

# randomly select who goes first

# instantiate game
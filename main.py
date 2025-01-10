import os
import numpy as np
import random
import sc2.ids
import sc2
from sc2.constants import *
from sc2 import run_game, Race, maps, Difficulty, Result
from sc2.player import Bot, Computer, Human
from sc2.ids import ability_id
from project_ai import Terran

def main():
    num_games = 80

    for game_number in range(0, 1):
        bot = Terran()
        run_game(
            maps.get("SiteDelta512V2AIE"),
            [Bot(Race.Terran, bot), Computer(Race.Terran, Difficulty.Medium)],
            realtime=True
        )
        bot.save_game_data(game_number)

if __name__ == "__main__":
    main()

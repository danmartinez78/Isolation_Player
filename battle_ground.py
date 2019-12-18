  
#!/usr/bin/env python
import traceback
from MM_agent import CustomPlayer as MM
from AB_agent import CustomPlayer as AB
from AB_agent import CustomPlayer as Lucy
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer
import platform
import random

if platform.system() != 'Windows':
    import resource
from time import time, sleep


def main():
    """AI BATTLE GROUND"""
    wins = 0
    losses = 0
    avg_turns = 0
    num_games = 10
    for i in range(num_games):
        print ("")
        # AI Lineup
        hp = HumanPlayer()
        r = RandomPlayer()
        L = Lucy()
        mm = MM()
        ab = AB()

        match_up = [L, hp]

        first = random.choice(match_up)
        if first == L:
            second = match_up[1]
            my_player = "CustomPlayer - Q1"
        else:
            second = L
            my_player = "CustomPlayer - Q2"

        game = Board(first, second, 7, 7)

        winner, move_history, termination = game.play_isolation(time_limit=1000, print_moves=False)
        display_moves = []
        for move_tup in move_history:
            for move in move_tup:
                move_to_append = list(move)
                display_moves.append(move_to_append[0:2])
        print (display_moves)
        #print "\n", winner, " has won. Reason: ", termination
        if winner == my_player:
            print ("\nLucy has won. Reason: ", termination)
            wins += 1
        else:
            print ("\nLucy lost. Reason: ", termination)
            losses += 1
        num_turns = len(move_history)
        avg_turns = round(float(avg_turns+num_turns)/(i+1))
        print ("\n game number:", i + 1)
        print ("\n turns:", num_turns)
        print ("\n average turns:", avg_turns)
        print ("\n win percent:", float(wins)/(i+1))
    # Uncomment to see game
    # print game_as_text(winner, move_history, termination, output_b)
    print ("\n Total win percent:", wins/float(num_games))
    print ("\n average turns:", avg_turns, "\n")

if __name__ == "__main__":
    main()
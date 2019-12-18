#!/usr/bin/env python
import math

from isolation import Board, game_as_text
from random import randint
import random
import copy
from pprint import pprint as pp

import os
import sys
import time
import functools
sys.path.append(os.path.join(os.path.dirname(__file__), './uArm-Python-SDK/'))
from uarm.wrapper import SwiftAPI

import perception

"""
@type game: (Board, boolean, string)
"""

class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.
        Note:
            1. Be very careful while doing opponent's moves. You might end up
               reducing your own moves.
            3. If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                param1 (Board): The board and game state.
                param2 (bool): True if maximizing player is active.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """
        my_moves = game.get_legal_moves()
        enemy_moves = game.get_opponent_moves()
        score = len(my_moves) - len(enemy_moves)
        return score


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_active=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        @type game: Board
        @type game.state: list
        """

        turn_number = game.move_count/2  # TODO: verify this

        center_row = game.height/2
        center_col = game.width/2

        # active_player = game.get_active_players_queen()[15:17]
        # inactive_player = game.get_inactive_players_queen()[15:17]

        if maximizing_player_active:
            my_moves = game.get_legal_moves()
            enemy_moves = game.get_opponent_moves()
            my_pos = game.__last_queen_move__[game.__active_players_queen__][0:2]
            enemy_pos = game.__last_queen_move__[game.__inactive_players_queen__][0:2]
            # my_pos = [(i, j.index(active_player)) for i, j in enumerate(game.get_state()) if active_player in j][0]
            # enemy_pos = [(i, j.index(inactive_player)) for i, j in enumerate(game.get_state()) if inactive_player in j][0]
        else:
            my_moves = game.get_opponent_moves()
            enemy_moves = game.get_legal_moves()
            enemy_pos = game.__last_queen_move__[game.__active_players_queen__][0:2]
            my_pos = game.__last_queen_move__[game.__inactive_players_queen__][0:2]
            # enemy_pos = [(i, j.index(active_player)) for i, j in enumerate(game.get_state()) if active_player in j][0]
            # my_pos = [(i, j.index(inactive_player)) for i, j in enumerate(game.get_state()) if inactive_player in j][0]

        # TODO: Tune penalties
        # TODO: Look for partitions!

        center_penalty = math.sqrt((my_pos[0] - center_row)**2 + (my_pos[1] - center_col)**2)

        proximity_penalty = -math.sqrt((my_pos[0] - enemy_pos[0])**2 + (my_pos[1] - enemy_pos[1])**2)

        bump_penalty = 0
        if any(move[2] is True for move in enemy_moves):
            bump_penalty = 5

        score = len(my_moves) - 2*len(enemy_moves) - center_penalty - proximity_penalty - bump_penalty
        # print game.print_board()
        # print "Max player active?", maximizing_player_active
        # print"My moves available:", len(my_moves)
        # print"Enemy moves available:", len(enemy_moves)
        # print"score", score
        return score


class CustomPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=30, eval_fn=OpenMoveEvalFn(), mode="alpha_beta"):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.mode = mode

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

            Note:
                1. Do NOT change the name of this 'move' function. We are going to call
                the this function directly.
                2. Change the name of minimax function to alphabeta function when
                required. Here we are talking about 'minimax' function call,
                NOT 'move' function name.
                Args:
                game (Board): The board and game state.
                legal_moves (dict): Dictionary of legal moves and their outcomes
                time_left (function): Used to determine time left before timeout

            Returns:
                tuple: best_move

            @type game: Board
            """
        selected_move = None
        if len(legal_moves) == 0:
            return None
        elif len(legal_moves) == 1:
            selected_move = legal_moves[0]

        turn_number = game.move_count
        if turn_number < 2:
            return random.choice(legal_moves)

        if self.mode == "minimax":
            best_move, utility = self.minimax(game, time_left, depth=self.search_depth, maximizing_player=True)
        else:
            best_move, utility = self.alphabeta(game, time_left, depth=self.search_depth, maximizing_player=True)

        selected_move = best_move
        # actuate move
        my_pos = game.__last_queen_move__[game.__active_players_queen__][0:2]
        enemy_pos = game.__last_queen_move__[game.__inactive_players_queen__][0:2]
        new_state = game[0].forecast_move(move)
        my_new_position = new_state.__last_queen_move__[game.__active_players_queen__][0:2]
        enemy_new_position = new_state.__last_queen_move__[game.__inactive_players_queen__][0:2]
        # if my enemy old and new dont match -> bump -> move enemy piece
        # move my piece

        return selected_move

    def opening_move(self, game, legal_moves, turn_number):
        if turn_number == 0:
            return 1, 1, False
        if turn_number == 1:
            if game.is_spot_open(1, 1):
                return 1, 1, False
            else:
                return game.height - 2, game.width - 2, False

    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory. """
        return self.eval_fn.score(game, maximizing_player)

    def rotate_board(self, game_board):
        rotated = zip(*game_board[::-1])
        rotated_list = []
        for row in rotated:
            rotated_list.append(list(row))
        return rotated_list

    def mirror_vert(self, game_board):
        mirror = []
        gb = copy.deepcopy(game_board)
        for row in game_board:
            row.reverse()
            mirror.append(row)
        #print game_board
        #print mirror
        return mirror

    def mirror_horiz(self, game_board):
        board_to_mirror = copy.deepcopy(game_board)
        board_to_mirror.reverse()
        return board_to_mirror

    def minimax(self, game, time_left, depth, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val

        @type game: Board
        """
        # TODO: test iterative deepening, quiescent search?
        # TODO: use additional values returned from forecast_moves
        # start with maximizing player
        moves = game.get_legal_moves()
        sorted_moves = self.sort_moves(moves)
        iter_depth = 0
        while time_left() > 50 and iter_depth < depth:
            iter_depth += 1
            best_val = float("-inf")
            completed_search = False
            for move in sorted_moves:  # sorted_moves:
                new_state = game.forecast_move(move)
                val, completed_search = self.min_choice(new_state, time_left, iter_depth - 1, False)
                if val > best_val:
                    best_move = move
                    best_val = val
            if completed_search:
                final_move = best_move
                final_val = best_val
        #     print "depth searched:", iter_depth
        #     print "search status:", completed_search
        #     print "-----------------best utility:", best_val
        #     print "-----------------best move:", best_move
        # print "+++++++++++++++++best utility:", final_val
        # print "+++++++++++++++++best move:", final_move
        return final_move, final_val

    def min_choice(self, game, time_left, depth, maximizing_player_active = False):
        """
            @type game: (Board, boolean, string)
        """
        # #print"\n---------------------------min"
        # print"\ndepth:", self.search_depth - depth
        # print game[0].print_board()
        moves = game[0].get_legal_moves()
        if len(moves) == 0:
            # min player out of moves!
            return float("inf"), True

        if game[1]:
            # max pushed min off board
            return float("inf"), True

        if depth == 0:
            val = self.utility(game[0], maximizing_player_active)
            return val, True

        if time_left() < 10:
            val = self.utility(game[0], maximizing_player_active)
            return val, False

        #printgame[0].print_board()
        #print"moves available:", len(moves)
        #for move in moves:
        #print"Enemy Moves:", move
        sorted_moves = self.sort_moves(moves)
        best_val = float("inf")
        for move in sorted_moves:
            new_state = game[0].forecast_move(move)
            val, completed_search = self.max_choice(new_state, time_left, depth - 1, maximizing_player_active=True)
            if val < best_val:
                best_val = val
        #print"done"
        return best_val, completed_search

    def max_choice(self, game, time_left, depth, maximizing_player_active = True):
        """
            @type game: (Board, boolean, string)
        """
        # print "\n---------------------------min"
        # print"\ndepth:", self.search_depth - depth
        # print game[0].print_board()
        moves = game[0].get_legal_moves()
        if len(moves) == 0:
            # max player out of moves!
            return float("-inf"), True

        if game[1]:
            # min pushed max off board
            return float("-inf"), True

        if depth == 0:
            val = self.utility(game[0], maximizing_player_active)
            return val, True

        if time_left() < 10:
            val = self.utility(game[0], maximizing_player_active)
            return val, False

        #printgame[0].print_board()
        #print"moves available:", len(moves)
        #for move in moves:
        #print"My Moves:", move
        sorted_moves = self.sort_moves(moves)
        best_val = float("-inf")
        for move in sorted_moves:
            new_state = game[0].forecast_move(move)
            val, completed_search = self.min_choice(new_state, time_left, depth - 1, maximizing_player_active=False)
            if val > best_val:
                best_val = val
        # print"done"
        return best_val, completed_search

    def sort_moves(self, move_list):
        # easy way to prioritize other moves? eval_function?
        # simple: sort push moves to top
        move_list.sort(key=lambda p: p[2])
        if move_list[-1][2]:
            move_list.insert(0, move_list.pop(-1))
        return move_list

    def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val

        @type game: Board
        """
        # TODO: quiescent search?
        # TODO: use additional values returned from forecast_moves
        # TODO: add some time, depth, nodes visited telemetry
        # start with maximizing player
        moves = game.get_legal_moves()
        sorted_moves = self.sort_moves(moves)
        iter_depth = 0
        while time_left() > 100 and iter_depth < depth:
            iter_depth += 1
            best_val = float("-inf")
            completed_search = False
            alpha = float("-inf")
            beta = float("inf")
            symmetry_table = []
            for move in sorted_moves:  # sorted_moves:
                new_state = game.forecast_move(move)

                board_state = new_state[0].get_state()
                if board_state in symmetry_table:
                    # print "SYMMETRY!"
                    # for row in board_state:
                    #     print row
                    continue

                val, completed_search = self.alpha_min(new_state, time_left, iter_depth - 1, alpha, beta, False)

                # print "move:\n"
                # for row in board_state:
                #     print row

                bs1 = copy.deepcopy(board_state)
                bs2 = copy.deepcopy(board_state)

                mv = self.mirror_vert(bs1)
                mh = self.mirror_horiz(bs2)
                mv_1r = self.rotate_board(mv)
                mv_2r = self.rotate_board(mv_1r)
                mv_3r = self.rotate_board(mv_2r)
                mh_1r = self.rotate_board(mh)
                mh_2r = self.rotate_board(mh_1r)
                mh_3r = self.rotate_board(mh_2r)

                symmetry_table.append(board_state)
                symmetry_table.append(mv)
                symmetry_table.append(mv_1r)
                symmetry_table.append(mv_2r)
                symmetry_table.append(mv_3r)
                symmetry_table.append(mh)
                symmetry_table.append(mh_1r)
                symmetry_table.append(mh_2r)
                symmetry_table.append(mh_3r)

                # print "move:\n"
                # for row in board_state:
                #     print row
                #
                # print "\nsymmetry table:\n"
                # print len(symmetry_table), "length\n"
                # for board in symmetry_table:
                #     for row in board:
                #         print row
                #     print "\n"
                # print "\n\n"

                if val > best_val:
                    best_move = move
                    best_val = val
                    if best_val > alpha:
                        alpha = best_val

            if completed_search:
                final_move = best_move
                final_val = best_val
        #     print "depth searched:", iter_depth
        #     print "search status:", completed_search
        #     print "-----------------best utility:", best_val
        #     print "-----------------best move:", best_move
        # print "+++++++++++++++++best utility:", final_val
        # print "+++++++++++++++++best move:", final_move
        return final_move, final_val

    def alpha_min(self, game, time_left, depth, alpha, beta, maximizing_player_active = False):
        """
            @type game: (Board, boolean, string)
        """
        # #print"\n---------------------------min"
        # print"\ndepth:", self.search_depth - depth
        # print game[0].print_board()
        moves = game[0].get_legal_moves()
        if len(moves) == 0:
            # min player out of moves!
            return float("inf"), True

        if game[1]:
            # max pushed min off board
            return float("inf"), True

        if depth == 0:
            val = self.utility(game[0], maximizing_player_active)
            return val, True

        if time_left() < 10:
            val = self.utility(game[0], maximizing_player_active)
            return val, False

        #printgame[0].print_board()
        #print"moves available:", len(moves)
        #for move in moves:
        #print"Enemy Moves:", move
        sorted_moves = self.sort_moves(moves)
        best_val = float("inf")
        for move in sorted_moves:
            new_state = game[0].forecast_move(move)
            val, completed_search = self.alpha_max(new_state, time_left, depth - 1, alpha, beta, maximizing_player_active=True)
            if val < best_val:
                best_val = val
                if best_val < alpha:
                    return best_val, True
                if best_val < beta:
                    beta = best_val
        #print"done"
        return best_val, completed_search

    def alpha_max(self, game, time_left, depth, alpha, beta, maximizing_player_active = True):
        """
            @type game: (Board, boolean, string)
        """
        # print "\n---------------------------min"
        # print"\ndepth:", self.search_depth - depth
        # print game[0].print_board()
        moves = game[0].get_legal_moves()
        if len(moves) == 0:
            # max player out of moves!
            return float("-inf"), True

        if game[1]:
            # min pushed max off board
            return float("-inf"), True

        if depth == 0:
            val = self.utility(game[0], maximizing_player_active)
            return val, True

        if time_left() < 10:
            val = self.utility(game[0], maximizing_player_active)
            return val, False


        #printgame[0].print_board()
        #print"moves available:", len(moves)
        #for move in moves:
        #print"My Moves:", move
        sorted_moves = self.sort_moves(moves)
        best_val = float("-inf")
        for move in sorted_moves:
            new_state = game[0].forecast_move(move)
            val, completed_search = self.alpha_min(new_state, time_left, depth - 1, alpha, beta, maximizing_player_active=False)
            if val > best_val:
                best_val = val
                if best_val > beta:
                    return best_val, True
                if best_val > alpha:
                    alpha = best_val

        # print"done"
        return best_val, completed_search

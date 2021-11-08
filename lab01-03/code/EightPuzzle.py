# -*- coding: utf-8 -*-
"""
Created on Sept 2021

Finished on Oct 25, 2021

@author: Songqing Zhao, Minzu University of China
"""
import math


def board_check(chess_board):
    """
    Check if board is legal
    :param chess_board:
    :return: return 1 means ok
             return 0 means no
    """
    return sorted(chess_board) == [x for x in range(0, 9)]


def print_board(chess_board):
    """
    Print standard board
    :param chess_board:
    :return: have no return
    """
    print(f'[{chess_board[0]}', end='')
    n = int(math.sqrt(len(chess_board) + 1))
    for i in range(0, n):
        for j in range(0, n):
            if i != 0 or j != 0:
                print(f' {chess_board[i * n + j]}', end='')
        if i != n - 1:
            print()
    if chess_board[-1] == 'end':
        print('] solution')
    else:
        print(']')


def move_chess(chess_board, movement, chess=0):
    """
    Just move chess
    :param chess_board: Current chess board, will not change
    :param movement: Many moves
    :param chess: 0
    :return: Current chess board after moves
    """
    output = chess_board[:]
    for move in movement:
        chess_index = output.index(int(chess))
        temp_move = output[chess_index]
        if move == 'l':
            output[chess_index] = output[chess_index - 1]
            output[chess_index - 1] = temp_move
        elif move == 'r':
            output[chess_index] = output[chess_index + 1]
            output[chess_index + 1] = temp_move
        elif move == 'u':
            output[chess_index] = output[chess_index - 3]
            output[chess_index - 3] = temp_move
        elif move == 'd':
            output[chess_index] = output[chess_index + 3]
            output[chess_index + 3] = temp_move
        else:
            return 0
    return output


class EightPuzzle:
    def __init__(self, beginning_state=None, goal_state=None):
        if beginning_state is None:
            self.beginning_state = \
                [2, 8, 3,
                 1, 6, 4,
                 7, 0, 5]
        else:
            self.beginning_state = beginning_state
        if goal_state is None:
            self.goal_state = \
                [1, 2, 3,
                 8, 0, 4,
                 7, 6, 5]
        else:
            self.goal_state = goal_state
        if not self.have_solution():
            print('Function has exited!')
            exit()
        self.all_boards = [beginning_state]
        self.default_move_order = ['l', 'r', 'u', 'd']
        self.temp_path = []

    def have_solution(self):
        """
        Check if the problem is solvable
        :return: return 1 means have solution
                 return 0 means no solution
        """
        for state in [self.beginning_state, self.goal_state]:
            if not board_check(state):
                print(state, 'is not legal! Check it!')
                return 0
        temp_beginning = self.beginning_state[:]
        temp_goal = self.goal_state[:]
        temp_beginning.remove(0)
        temp_goal.remove(0)
        num = 0
        for i in range(1, len(temp_beginning)):
            for j in range(0, i):
                if temp_beginning[i] > temp_beginning[j]:
                    num += 1
                if temp_goal[i] > temp_goal[j]:
                    num += 1
        if num % 2:
            print('No solution at all!')
        return (num + 1) % 2

    def move_feasible(self, chess_board, move, chess=0):
        """
        Check if can move. if can then move
        :param chess_board: Current chess board, will not change
        :param move: Single move
        :param chess: 0
        :return: Current chess board after move
        """
        if (move == 'u' and chess_board.index(int(chess)) - 3 >= 0) or \
                (move == 'd' and chess_board.index(int(chess)) + 3 <= 8) or \
                (move == 'l' and chess_board.index(int(chess)) % 3 != 0) or \
                (move == 'r' and chess_board.index(int(chess)) % 3 != 2):
            temp_move = move_chess(chess_board[:], move)
            if temp_move not in self.all_boards:
                self.all_boards.append(temp_move[:])
                return temp_move
        return 0

    def evaluation_function(self, input_state):
        score = 0
        for i in range(0, 9):
            if input_state[i] == self.goal_state[i] and input_state[i]:
                score += 1
        return score

    def change_move_order(self, input_state, flag=None, chess=0):
        if flag is None:
            return self.default_move_order
        current_score = self.evaluation_function(input_state[:])
        move_order = {}
        for move in self.default_move_order:
            if (move == 'u' and input_state.index(int(chess)) - 3 >= 0) or \
                    (move == 'd' and input_state.index(int(chess)) + 3 <= 8) or \
                    (move == 'l' and input_state.index(int(chess)) % 3 != 0) or \
                    (move == 'r' and input_state.index(int(chess)) % 3 != 2):
                score = self.evaluation_function(move_chess(input_state[:], move))
                if score < current_score and flag == 'p':
                    continue
                move_order[move] = score
        output = list({k: v for k, v in sorted(move_order.items(), key=lambda item: item[1])}.keys())[::-1]
        return output

    def generate_depth_path(self, chess_board, used_path, depth=4, record_path=None, flag=None):
        """
        Generate the depth path
        :param chess_board: Current chess board, will not change
        :param used_path: Used to change beginning state to curren state, will not change. \
                          Only used as template when adding things in record\
                          Just a list including one used path list
        :param depth: Depth of search
        :param record_path: Output storage, will append new path that is based on path
        :param flag:
        :return: no return
        """
        if record_path is None:
            record_path = self.temp_path
        move_order = self.change_move_order(chess_board, flag)
        if depth > 0 and chess_board != self.goal_state:
            for move in move_order:
                temp_move = self.move_feasible(chess_board, move)
                if temp_move != 0:
                    output = used_path[:]
                    output.append(move)
                    self.generate_depth_path(temp_move[:], output, depth - 1, record_path, flag)
        else:
            temp = used_path[:]
            if chess_board == self.goal_state:
                temp.append('end')
                record_path.append(temp)
                # print('Solution: ', path)
            else:
                record_path.append(temp)

    def search(self, method, para=2, upper=150, flag=None, chess_board=None, input_path=None, aim_solution_num=1):
        """
        blind_search
        :param method: BFS, DFS
        :param para: Width or Depth
        :param upper: Iter upper num
        :param flag:
        :param chess_board: Always beginning board
        :param input_path: Current solutions
        :param aim_solution_num: Expected answer num
        :return: All path that have been used
        """
        if (method.upper() == 'BFS' or 'DFS') or (method.lower() == 'pruning' or 'a_star'):
            if method.upper() != 'BFS':
                width = 0
                depth = para
            else:
                width = para
                depth = 1
        else:
            return 0
        if chess_board is None:
            chess_board = self.beginning_state
        if input_path is None or input_path == []:
            input_path = []
            self.generate_depth_path(chess_board, [], 1, input_path, flag)
            beginning_num = len(input_path)
        solution_num = 0
        iter_list = input_path[:]
        output = []
        for i_path in iter_list:
            if method != 'BFS' or len(i_path) <= width or i_path[-1] == 'end':
                output.append(i_path)
                if i_path[-1] != 'end' and iter_list.index(i_path) < upper:
                    self.temp_path.clear()
                    self.generate_depth_path(move_chess(chess_board[:], i_path[:]), i_path[:], depth, None, flag)
                    for temp in range(0, len(self.temp_path)):
                        iter_list.append(self.temp_path.pop())
                else:
                    if i_path[-1] == 'end':
                        solution_num += 1
                        print(method.title(), 'Solution: ', i_path)
                    if iter_list.index(i_path) >= upper:
                        print(f'Iteration number {upper} hits the upper!')
                    if solution_num >= aim_solution_num or iter_list.index(i_path) >= upper:
                        self.temp_path.clear()
                        if method.upper() != 'BFS':
                            output = output[beginning_num:]
                        return output
            else:
                print(f'Width number {width} hits the setting!')
                return output

    def self_test(self, method, para=4, upper=150, flag=None):
        """
        test
        :param method:
        :param para:
        :param upper:
        :return:
        """
        search_result = self.search(method, para, upper, flag)
        print(search_result)
        print(f'{method} tried {len(search_result)} times.\n')
        self.all_boards.clear()

    # def heuristic_search(self, method, flag='p', para=4, upper=150, chess_board=None, input_path=None,
        # aim_solution_num=1):
        # """
        # blind_search
        # :param method: BFS, DFS
        # :param flag: 'p' means pruning
        # :param para: Width or Depth
        # :param upper: Iter upper num
        # :param chess_board: Always beginning board
        # :param input_path: Current solutions
        # :param aim_solution_num: Expected answer num
        # :return: All path that have been used
        # """
    #     if method.lower() == 'pruning' or 'a_star':
    #         depth = para
    #     else:
    #         return 0
    #     if chess_board is None:
    #         chess_board = self.beginning_state
    #     if input_path is None or input_path == []:
    #         input_path = []
    #         self.generate_depth_path(chess_board, [], 1, input_path, flag)
    #         beginning_num = len(input_path)
    #     solution_num = 0
    #     iter_list = input_path[:]
    #     output = []
    #     for i_path in iter_list:
    #         output.append(i_path)
    #         if i_path[-1] != 'end' and iter_list.index(i_path) < upper:
    #             self.temp_path.clear()
    #             self.generate_depth_path(move_chess(chess_board[:], i_path[:])[:], i_path[:], depth, None, flag)
    #             for temp in range(0, len(self.temp_path)):
    #                 iter_list.append(self.temp_path.pop())
    #         else:
    #             if i_path[-1] == 'end':
    #                 solution_num += 1
    #                 print(method, 'Solution: ', i_path)
    #             if iter_list.index(i_path) >= upper:
    #                 print(f'Iteration number {upper} hits the upper!')
    #             if solution_num >= aim_solution_num or iter_list.index(i_path) >= upper:
    #                 self.temp_path.clear()
    #                 return output[beginning_num:]
    #
    # def self_test_heuristic(self, method, flag='p', para=4, upper=150):
    #     """
    #     test
    #     :param method:
    #     :param flag:
    #     :param para:
    #     :param upper:
    #     :return:
    #     """
    #     blind_search_result = self.heuristic_search(method, flag, para, upper)
    #     print(blind_search_result)
    #     print(f'{method.title()} method tried {len(blind_search_result)} times.\n')
    #
    #     self.all_boards.clear()
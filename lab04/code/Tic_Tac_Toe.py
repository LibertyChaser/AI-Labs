import random

class Tic_Tac_Toe:
    def __init__(self, level=9):
        self.WIN = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                    (0, 3, 6), (1, 4, 7), (2, 5, 8),
                    (0, 4, 8), (2, 4, 6))
        self.row = ((0, 1, 2), (3, 4, 5), (6, 7, 8))

        self.X_PLAYER = -1
        self.EMPTY = 0
        self.O_AI = 1
        self.RESULT = ['Draw!', 'AI Win!', 'Player Win!']
        self.MARK = ['·', 'O', 'X']

        self.level = level

    def isEmpty(self, board):
        for item in range(0, 9):
            if board[item] == self.EMPTY:
                return True
        return False

    def check_winner(self, input_board):
        """
        Determine whether a winner has been generated
        :param input_board:
        :return:
            Return -1 means the player wins, 1 means the ai wins, 0 means a tie or not yet over
            Return 0 in the while loop of the main program to indicate that it is not over yet,
            and return 0 after the end of the while loop to indicate a tie
        """
        for i in self.WIN:
            if input_board[i[0]] == input_board[i[1]] == input_board[i[2]] == 1:
                return self.O_AI
            elif input_board[i[0]] == input_board[i[1]] == input_board[i[2]] == -1:
                return self.X_PLAYER
        return 0

    def print_board(self, input_board):
        """

        :param input_board:
        :return:
        """
        for i in self.row:
            output = ' '
            for j in i:
                output += self.MARK[input_board[j]] + ' '
            print(output)

    # Determine the next step of the computer, use the alpha-bate pruning strategy
    def move(self, input_board, level=9):
        board1 = input_board
        best = -2
        ai_move_record = []
        for i in range(0, 9):
            if board1[i] == self.EMPTY:
                board1[i] = self.O_AI

                val = self.A_B(board1, self.X_PLAYER, self.O_AI, -2, 2, level)
                board1[i] = self.EMPTY
                if val > best:
                    best = val
                    ai_move_record = [i]
                elif val == best:
                    ai_move_record.append(i)
        return random.choice(ai_move_record)

    def A_B(self, input_board, player, next_player, alpha, beta, level=9):
        """
        The search depth here is 9, and the valuation function value of the leaf node is 1 for the ai to win,
        -1 for the player to win, and 0 for a tie.
        :param input_board:
        :param player:
        :param next_player:
        :param alpha:
        :param beta:
        :param level:
        :return:
        """
        board1 = input_board
        win = self.check_winner(board1)
        if win != self.EMPTY:
            return win
        elif not self.isEmpty(board1):
            return 0

        for move_choice in range(0, 9):
            if board1[move_choice] == self.EMPTY:
                board1[move_choice] = player
                val = self.A_B(board1, next_player, player, alpha, beta, level)
                board1[move_choice] = self.EMPTY

                if player == self.O_AI:
                    alpha = max(alpha, val)
                    if alpha >= beta:
                        return beta
                else:
                    beta = min(beta, val)
                    if beta <= alpha:
                        return alpha
        if player == self.O_AI:
            re = alpha
        else:
            re = beta
        return re

    def Begin(self):
        first = input("Please choose which side to play first.\n"
                      "Input 'X' means the player will play first.\n"
                      "Input 'O' means the ai will play first:\n")
        if first.lower() == "x":
            next_move = self.X_PLAYER
        elif first.lower() == "o":
            next_move = self.O_AI
        else:
            print("The input is wrong! Player will play first by default.")
            next_move = self.X_PLAYER

        board = [self.EMPTY for _ in range(9)]

        while self.isEmpty(board) and not self.check_winner(board):
            if next_move == self.X_PLAYER:
                self.print_board(board)
                while True:
                    try:
                        player_move = int(input("Please enter the position you want to place (0-8):\n"))
                        if board[player_move] != self.EMPTY:
                            print('Wrong location selection. Please select again!')
                            continue
                        break
                    except:
                        print("Input error, please try again")
                        continue
                board[player_move] = self.X_PLAYER
                next_move = self.O_AI
            if next_move == self.O_AI and self.isEmpty(board):
                computer_move = self.move(board, self.level)
                board[computer_move] = self.O_AI
                next_move = self.X_PLAYER

        self.print_board(board)
        print(self.RESULT[self.check_winner(board)])

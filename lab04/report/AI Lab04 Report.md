# AI Lab04 Report

---

[toc]

Author: Songqing Zhao, Minzu University of China 

Written at Nov 2^nd^, 2021

> 

---

## Lab Purpose

Alpha-beta pruning solves the problem of one-character chess game

Game: tic-tac-toe (tic-tac-toe)

Two players, one max and one min, put X or O each time, alternately walking until the end state.

The definition of winning: There are 3 Xs or 3 Os on the board, in the same row or column or diagonal.

End state: max wins or min wins, or the board is full of draws.

Use alpha-beta pruning strategy to realize the game of tic-tac-toe.

## Lab Principle

### Analysis of alpha-beta pruning algorithm

In the search algorithm optimization, pruning is to avoid some unnecessary traversal processes through certain judgments. To put it vividly, it is to pruning some of the "branches" in the search tree, so it is called pruning. The core problem of applying pruning optimization is to design a pruning judgment method, that is, a method to determine which branches should be discarded and which branches should be retained.

### Three principles of pruning optimization:

Correct, accurate, and efficient

1) Correctness
Branches can not be cut like you like. If you just cut off the branch with the best solution, the pruning will lose its meaning. Therefore, the premise of pruning is to ensure that it is not lost Correct result.
2) Accuracy
    On the basis of ensuring the correctness, it is necessary to analyze specific problems and adopt appropriate judgment methods to make as many branches that do not contain the optimal solution be cut as much as possible to achieve the purpose of "optimization" of the program. It can be said that , The accuracy of pruning is a measure of the quality of an optimization algorithm.
3) High efficiency
The fundamental purpose of designing an optimization program is to reduce the number of searches and reduce the running time of the program. But in order to reduce the number of searches as much as possible, we must spend time designing an optimized algorithm with higher accuracy. The accuracy of the increase, the number of judgments must increase, which in turn leads to an increase in time-consuming, which leads to contradictions. Therefore, how to find a balance between optimization and efficiency, so that the time complexity of the program is reduced as much as possible , Is also very important. If the judgment effect of a pruning is very good, but it takes a lot of time to judge and compare, the result is that the entire program runs the same as the one that has not been optimized, so it is too costly.



## Lab Procedure

### Initialization of Tic_Tac_Toe

The position of the chessboard is represented by numbers 0-8, and the corresponding use list is stored;

Win tuples store all possible winning positions, and position combinations are also represented by tuples.
(4

In order to output the chessboard, set the mark list [‘·’,’O’,’X’] to convert the used -1, 1, 0 into a mark symbol through a loop.

```python
import random
```

```python
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
```

Judge whether there are any vacancies on the current board, true means there are still empty, false means there are n’t empty.

Judge whether each position of the current board is empty

```python
    def isEmpty(self, board):
        for item in range(0, 9):
            if board[item] == self.EMPTY:
                return True
        return False
```

Determine whether a winner has been generated.

```python
    def check_winner(self, input_board):
        """
        :param input_board:
        :return:
            -1 means the player wins, 
            1 means the ai wins, 
            0 means a tie or not yet over
            	Return 0 in the while loop of the main program to indicate that it is not over yet,
            	and return 0 after the end of the while loop to indicate a tie
        """
        for i in self.WIN:
            if input_board[i[0]] == input_board[i[1]] == input_board[i[2]] == 1:
                return self.O_AI
            elif input_board[i[0]] == input_board[i[1]] == input_board[i[2]] == -1:
                return self.X_PLAYER
        return 0
```

```python
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
```

Determine the next step of the computer, use the alpha-bate pruning strategy

```python
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
```

### Implementation of α-β pruning rule

- Search strategy: K-step game; depth first; expand one node at a time; one side
  Extended side evaluation
- For an AND node, it takes the smallest backward value in the current child node as
  It reverses the upper bound of the value, which is called β value (β <= minimum value) 
-  For an OR node, it takes the maximum reverse value in the current child node as
  It reverses the lower bound of the value and calls this the alpha value (α>=maximum value)
- α: The lower bound of the estimated value of the Max node
- β: Upper bound of the estimated value of Min node

The search depth, that is, the number of steps deduced down is 9, the leaf node evaluation function is defined as f(board)=-1 for the human player to win, f(board)=1 for the computer to win, and f(board)=0 for the tie.


```python
    def A_B(self, input_board, player, next_player, alpha, beta, level=9):
        """
        The search depth here is 9.
        The valuation function value of the leaf node is 1 for the ai to win,
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
```

### Begin function


```python
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

```

## Lab Result

### Test code

In `main.py`

```python
import Tic_Tac_Toe

if __name__ == '__main__':
    Tic_Tac_Toe.Tic_Tac_Toe(9).Begin()

```

### Python console result

Run `main.py`

```python
Please choose which side to play first.
Input 'X' means the player will play first.
Input 'O' means the ai will play first:
 · · · 
 · · · 
 · · · 
Please enter the position you want to place (0-8):
 · · · 
 · X · 
 · · O 
Please enter the position you want to place (0-8):
```

To test the input checkin function,  I put a existed number 4, and the program show the tip.

```python
Wrong location selection. Please select again!
Please enter the position you want to place (0-8):
 · · · 
 O X X 
 · · O 
Please enter the position you want to place (0-8):
Wrong location selection. Please select again!
Please enter the position you want to place (0-8):
 · · X 
 O X X 
 O · O 
Please enter the position you want to place (0-8):
 X · X 
 O X X 
 O O O 
```

Winner check function works well.

```python
AI Win!
```

## Improvement and innovation

AI is too strong.

Don’t have the method to set depth, since the problem don’t worth it.
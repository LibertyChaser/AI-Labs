# AI Lab01-03 Report

---

[toc]

Author: Songqing Zhao, Minzu University of China

Written at Oct 25^th^, 2021

> [P1379 八数码难题](https://www.luogu.com.cn/problem/P1379)

---

## Lab Purpose

### Lab01-02

Use state space method to solve eight digital puzzles, including these search methods:

1. Depth first search(DFS)
   - The search tree generated by depth-first search sets the depth limit to 4
2. Breadth first search(BFS)

### Lab03

Use the state space method to solve the eight-digit puzzle including 2 heuristic search algorithms

## Lab Principle

On a 3×3 chessboard, there are 8 chess pieces, and each chess piece is marked with a number from 1 to 8. There is a space in the chessboard, and the space is represented by 0. The pieces around the space can be moved into the space. 

The problem to be solved is: give an initial layout (initial state) and target layout (in order to make the problem simple, set the target state as 123804765), find a moving method with the fewest steps, and realize the transition from the initial layout to the target layout 

### Principle of breadth first search
1. Definition
If the search is to expand the nodes in order to the extent close to the starting node, then this kind of search is called a breadth-first search.
2. Features
This kind of search is carried out layer by layer. Before searching for any node in the next layer, all the nodes in this layer must be searched.
There are nodes.
3. Breadth-first search algorithm
   1. Put the initial state node in the 0PEN table.
   2. If OPEN is an empty list, there is no solution and exit with failure; otherwise, continue.
   3. Move the first node (node ​​n) out of the OPEN list and put it in the CLOSED list.
   4. Expand node n. If there is no successor node, go to step ② above.
   5. Put all the successor nodes of node n to the end of the OPEN list, and provide pointers from these successor nodes back to n.
   6. If any successor node of node n is the target state node, find a solution and exit successfully; otherwise, go to step ②.

### Principle of Depth First Search
1. Definition
In this search, the newly generated (that is, the deepest) node is first expanded, and nodes with the same depth can be arranged arbitrarily. This kind of search is called depth first search.
2. Features
First, the result of expanding the deepest node makes the search go down from the starting node along a single path in the state space; only when the search reaches a state without descendants, it considers another alternative path.
3. Depth limit
In order to avoid considering a too long path (to prevent the search process from expanding along an unhelpful path), a maximum depth limit for node expansion is often given. If any nodes reach the depth limit, they will be treated as no successor nodes. My program settings
4. Depth first search algorithm
    1. Put the initial state node in the 0PEN table.
    2. If 0PEN is an empty table, there is no solution, and it fails to exit; otherwise, continue.
    3. Move the first node (node ​​n) out of the OPEN list and put it in the CLOSED list.
    4. Whether the depth of node n is equal to the maximum depth, turn to step ②; otherwise, continue.
    5. Expand node n. If there is no successor node, go to step ② above.
    6. Put all the successor nodes of node n to the front end of the OPEN table, and provide pointers from these successor nodes back to n.
    7. If any successor node of node n is the target state node, find a solution and exit successfully; otherwise, go to step ②.

## Lab Procedure

### Initialization of the eight-digit puzzle state space

#### Design state space representation
The space is defined as 0. The numbers are stored in the list by line. `beginning_state` is used to store the initial state nodes of the eight-digit puzzle. `goal_state` is used to store the target state node of the eight-digit puzzle. The movement of numbers in the eight-digit problem is regarded as the movement of spaces (0).

The initial state node will generate by the function called `generate_depth_path`, and this is the root node of the search tree. The algorithm needs to find one or more paths from the root node to the goal state node goal.

So we can create a class called `EightPuzzle` and define an initial function. `beginning_state` and `goal_state` set the same as what the ppt given.

```python
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
```

#### Design a set of operators
The operator used to solve this problem can be described by 4 rules: space (0) moves up, space (0) moves down, space (0) moves to the left, and space (0) moves to the right.

Note: The moving direction of the space must have numbers and the state generated after the movement is the state generated for the first time.
These four rules are in fact the way the search tree generates the next node from the current node.

#### How to move and the conditions for moving

1. Move up and down

The picture shows a node in the eight-digit puzzle solving process and its corresponding subscript in the list. Assuming that the space (0) is in the middle, then the corresponding subscript of the space (0) in the list is 4, if the space is (0) To move up or down, it essentially exchanges positions with 8 or 6.

2. Move left and right

In the same situation as above, if the space (0) is to be moved to the left or to the right, it will essentially exchange positions with 1 or 4

```python
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
```

#### Use conditions to check feasible

The list index of a space (0) differs from the list index of the upper and lower positions by 3

The list subscript after the space (0) is moved up or down still needs to be within the subscript range of the entire list

The list index of a space (0) differs from the list index of the left and right positions by 1

When the list subscript of a space (0) is not divisible by 3, it can be shifted to the left; when it is divisible by 3, it can be shifted to the right.

And we can’t move the chess when the state has already existed.

```python
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
```

### Judging whether there is a solution to the eight-digit problem

For any piece c[i] in the sequence of pieces, if j> i and c[j] <c[i], then  c[j] is a reverse order of c[i], or c[i] and c[ j] forms a reverse pair. Define the reverse number of chess pieces c[i] as the reverse number of c[i], and the reverse number of the chess piece sequence is the sum of the reverse numbers of all the chess pieces in the sequence.

Theorem: If a sequence of pieces is exchanged for n times of adjacent pieces, if n is an even number, the inverse ordinal parity of the sequence remains unchanged; if n is an odd number, the inverse ordinal number of the sequence will undergo parity mutual change.

Promotion:

1. When the reverse order of the number of pieces in the initial state of the chess game is odd, the eight-digit problem has no solution;
2. When the reverse order of the number of pieces in the initial state of the game is even, the eight-digit problem has a solution.

```python
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
```

### Blind search

#### Create a search path(Old version)

```python
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
        if flag is None:
            move_order = self.default_move_order
        elif flag == 'a':
            move_order = self.change_move_order(chess_board)
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
```

New version is blow:

```python
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
```



#### Blind search main function(Old version)

```python
    def blind_search(self, method, para=2, upper=150, chess_board=None, input_path=None, aim_solution_num=1):
        """
        blind_search
        :param method: BFS, DFS
        :param para: Width or Depth
        :param upper: Iter upper num
        :param chess_board: Always beginning board
        :param input_path: Current solutions
        :param aim_solution_num: Expected answer num
        :return: All path that have been used
        """
        method = method.upper()
        if method == 'BFS' or method == 'DFS' or method == 'A_star':
            if method == 'DFS':
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
            self.generate_depth_path(chess_board, [], 1, input_path)
        solution_num = 0
        iter_list = input_path[:]
        output = []

        for i_path in iter_list:
            if method != 'BFS' or len(i_path) <= width or i_path[-1] == 'end':
                output.append(i_path)
                if i_path[-1] != 'end' and iter_list.index(i_path) < upper:
                    self.temp_path.clear()
                    self.generate_depth_path(move_chess(chess_board[:], i_path[:]), i_path[:], depth)
                    for temp in range(0, len(self.temp_path)):
                        iter_list.append(self.temp_path.pop())
                else:
                    if i_path[-1] == 'end':
                        solution_num += 1
                        print(method, 'Solution: ', i_path)
                    if iter_list.index(i_path) >= upper:
                        print(f'Iteration number {upper} hits the upper!')
                    if solution_num >= aim_solution_num or iter_list.index(i_path) >= upper:
                        # all_depth_path = iter_list[:]
                        self.temp_path.clear()
                        if method == 'DFS':
                            output = output[4:]
                        return output
            else:
                print(f'Width number {width} hits the setting!')
                return output
```

New version is [here](#new-integrated-search-method). Besides, blind search and heuristic search have been integrated into one test code

### Heuristic search

#### Evaluation function

```python
    def evaluation_function(self, input_state):
        score = 0
        for i in range(0, 9):
            if input_state[i] == self.goal_state[i] and input_state[i]:
                score += 1
        return score
```

#### Change move order(Old version)

```python
    def change_move_order(self, input_state, chess=0):
        current_score = self.evaluation_function(input_state[:])
        move_order = {}
        for move in self.default_move_order:
            if (move == 'u' and input_state.index(int(chess)) - 3 >= 0) or \
                    (move == 'd' and input_state.index(int(chess)) + 3 <= 8) or \
                    (move == 'l' and input_state.index(int(chess)) % 3 != 0) or \
                    (move == 'r' and input_state.index(int(chess)) % 3 != 2):
                score = self.evaluation_function(move_chess(input_state[:], move))
                if score < current_score:
                    continue
                move_order[move] = score
        output = list({k: v for k, v in sorted(move_order.items(), key=lambda item: item[1])}.keys())[::-1]
        return output
```

New version is below

```python
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
```

#### Heuristic search main function(Old version)

```python
    def heuristic_search(self, method, para=4, upper=150, chess_board=None, input_path=None, aim_solution_num=1):
        """
        blind_search
        :param method: BFS, DFS
        :param para: Width or Depth
        :param upper: Iter upper num
        :param chess_board: Always beginning board
        :param input_path: Current solutions
        :param aim_solution_num: Expected answer num
        :return: All path that have been used
        """
        global beginning_num
        method = method.lower()
        if method == 'local_optima' or 'a_star':
            depth = para
        else:
            return 0
        if chess_board is None:
            chess_board = self.beginning_state
        if input_path is None or input_path == []:
            input_path = []
            self.generate_depth_path(chess_board, [], 1, input_path, 'a')
            beginning_num = len(input_path)
        solution_num = 0
        iter_list = input_path[:]
        output = []

        for i_path in iter_list:
            output.append(i_path)
            if i_path[-1] != 'end' and iter_list.index(i_path) < upper:
                self.temp_path.clear()
                self.generate_depth_path(move_chess(chess_board[:], i_path[:])[:], i_path[:], depth, None, 'a')
                for temp in range(0, len(self.temp_path)):
                    iter_list.append(self.temp_path.pop())
            else:
                if i_path[-1] == 'end':
                    solution_num += 1
                    print(method, 'Solution: ', i_path)
                if iter_list.index(i_path) >= upper:
                    print(f'Iteration number {upper} hits the upper!')
                if solution_num >= aim_solution_num or iter_list.index(i_path) >= upper:
                    self.temp_path.clear()
                    return output[beginning_num:]
```

New version is same as the blind search. And the method also renamed as `search`. New version is [here](#new-integrated-search-method)

### New integrated search method

```python
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
```

### Test Utilitis(Old version)

#### Blind search self test

```python
    def self_test_blind(self, method, para=4, upper=150):
        """
        test
        :param method:
        :param para:
        :param upper:
        :return:
        """
        blind_search_result = self.blind_search(method, para, upper)
        print(blind_search_result)
        print(f'Using, {method} tried {len(blind_search_result)} times.\n')
        self.all_boards.clear()
```

#### Heuristic search self test

```python
    def self_test_heuristic(self, method, para=4, upper=150):
        """
        test
        :param method:
        :param para:
        :param upper:
        :return:
        """
        blind_search_result = self.heuristic_search(method, para, upper)
        print(blind_search_result)
        print(f'{method.title()} method tried {len(blind_search_result)} times.\n')

        self.all_boards.clear()
```

These two methods have been integrated into one test code:

#### New test code

```python
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
```

### Other Auxiliary function

#### Check if it range from 0 to 8

```python
def board_check(chess_board):
    """
    Check if board is legal
    :param chess_board:
    :return: return 1 means ok
             return 0 means no
    """
    return sorted(chess_board) == [x for x in range(0, 9)]

```

#### Print state beautifully

```python
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
```

## Lab Result

### Test code

```python
from EightPuzzle import *

if __name__ == '__main__':

    beginning_state = \
        [2, 8, 3,
         1, 6, 4,
         7, 0, 5]

    goal_state = \
        [1, 2, 3,
         8, 0, 4,
         7, 6, 5]

    print('Beginning State:')
    print_board(beginning_state)
    print('Goal State:')
    print_board(goal_state)

    # Exp01 02
    DFS = EightPuzzle(None, goal_state).self_test('DFS', 4, 100)
    BFS = EightPuzzle(None, goal_state).self_test('BFS', 5, 100)

    # Exp03
    A_star = EightPuzzle(None, goal_state).self_test('a_star', 4, 100, 'a')
    pruning = EightPuzzle(None, goal_state).self_test('pruning', 4, 100, 'p')
    # a_star = EightPuzzle(None, goal_state).self_test_heuristic('a_star', 'a')
    # pruning = EightPuzzle(None, goal_state).self_test_heuristic('pruning')

```

### Display the running results

```python
Beginning State:
[2 8 3
 1 6 4
 7 0 5]
Goal State:
[1 2 3
 8 0 4
 7 6 5]
Dfs Solution:  ['u', 'u', 'l', 'd', 'r', 'end']
[['l', 'u', 'u', 'r', 'd'], ['l', 'u', 'u', 'r', 'r'], ['l', 'u', 'r', 'd', 'r'], ['l', 'u', 'r', 'd', 'l'], ['l', 'u', 'r', 'u', 'r'], ['l', 'u', 'r', 'u', 'l'], ['l', 'u', 'r', 'r', 'd'], ['l', 'u', 'r', 'r', 'u'], ['r', 'u', 'u', 'l', 'd'], ['r', 'u', 'u', 'l', 'l'], ['r', 'u', 'l', 'd', 'r'], ['r', 'u', 'l', 'd', 'l'], ['r', 'u', 'l', 'u', 'r'], ['r', 'u', 'l', 'u', 'l'], ['r', 'u', 'l', 'l', 'd'], ['r', 'u', 'l', 'l', 'u'], ['u', 'u', 'r', 'd', 'd'], ['u', 'u', 'r', 'd', 'l'], ['u', 'u', 'l', 'd', 'd'], ['u', 'u', 'l', 'd', 'r', 'end']]
DFS tried 20 times.

Bfs Solution:  ['u', 'u', 'l', 'd', 'r', 'end']
[['l'], ['r'], ['u'], ['l', 'u'], ['l', 'r'], ['r', 'u'], ['u', 'u'], ['u', 'r'], ['u', 'l'], ['l', 'u', 'u'], ['l', 'u', 'r'], ['r', 'u', 'u'], ['r', 'u', 'l'], ['u', 'u', 'r'], ['u', 'u', 'l'], ['u', 'r', 'd'], ['u', 'r', 'u'], ['u', 'l', 'd'], ['u', 'l', 'u'], ['l', 'u', 'u', 'r'], ['l', 'u', 'r', 'd'], ['l', 'u', 'r', 'u'], ['l', 'u', 'r', 'r'], ['r', 'u', 'u', 'l'], ['r', 'u', 'l', 'd'], ['r', 'u', 'l', 'u'], ['r', 'u', 'l', 'l'], ['u', 'u', 'r', 'd'], ['u', 'u', 'l', 'd'], ['u', 'r', 'd', 'l'], ['u', 'r', 'u', 'l'], ['u', 'l', 'd', 'r'], ['u', 'l', 'u', 'r'], ['l', 'u', 'u', 'r', 'd'], ['l', 'u', 'u', 'r', 'r'], ['l', 'u', 'r', 'd', 'r'], ['l', 'u', 'r', 'd', 'l'], ['l', 'u', 'r', 'u', 'r'], ['l', 'u', 'r', 'u', 'l'], ['l', 'u', 'r', 'r', 'd'], ['l', 'u', 'r', 'r', 'u'], ['r', 'u', 'u', 'l', 'd'], ['r', 'u', 'u', 'l', 'l'], ['r', 'u', 'l', 'd', 'r'], ['r', 'u', 'l', 'd', 'l'], ['r', 'u', 'l', 'u', 'r'], ['r', 'u', 'l', 'u', 'l'], ['r', 'u', 'l', 'l', 'd'], ['r', 'u', 'l', 'l', 'u'], ['u', 'u', 'r', 'd', 'd'], ['u', 'u', 'r', 'd', 'l'], ['u', 'u', 'l', 'd', 'd'], ['u', 'u', 'l', 'd', 'r', 'end']]
BFS tried 53 times.

A_Star Solution:  ['u', 'u', 'l', 'd', 'r', 'end']
[['u', 'r', 'u', 'l', 'd'], ['u', 'r', 'u', 'l', 'l'], ['u', 'r', 'd', 'l', 'l'], ['u', 'r', 'd', 'l', 'u'], ['u', 'l', 'd', 'r', 'r'], ['u', 'l', 'd', 'r', 'u'], ['u', 'l', 'u', 'r', 'r'], ['u', 'l', 'u', 'r', 'd'], ['u', 'u', 'r', 'd', 'd'], ['u', 'u', 'r', 'd', 'l'], ['u', 'u', 'l', 'd', 'd'], ['u', 'u', 'l', 'd', 'r', 'end']]
a_star tried 12 times.

Pruning Solution:  ['u', 'u', 'l', 'd', 'r', 'end']
[['u', 'l', 'u', 'r', 'd'], ['u', 'u', 'l', 'd', 'r', 'end']]
pruning tried 2 times.
```

Appearnly, heuristic search methods are much better than the blind search methods.

## Improvement and innovation

### Blind Search

1. For DFS, the depth can be adjusted manually. 
   - This is above from the subject setting requirements. So we can compare the productivity of the different settings
2. For BFS, the depth can be adjusted manually.

### Heuristic Search

1. Create an evaluation function to assess path value
2. Use pruning mathod to improve efficiency

### Others

1. Set the maximum number of iterations
2. Blind search and heuristic search have been integrated into one test code.

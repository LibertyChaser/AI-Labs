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

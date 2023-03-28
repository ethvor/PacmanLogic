# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def sentence3():
    """Using the symbols PacmanAlive[1], PacmanAlive[0], PacmanBorn[0], and PacmanKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** BEGIN YOUR CODE HERE ***"
    cnfsentence = to_cnf(sentence)
    solutionValue = pycoSAT(cnfsentence)

    if solutionValue:
        return solutionValue

    else:
        return False
    "*** END YOUR CODE HERE ***"


def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
  A = PropSymbolExpr('A');
  B = PropSymbolExpr('B');
 symbols = [A, B]
  atleast1 = atLeastOne(symbols)
    model1 = {A:False, B:False}
   print(pl_true(atleast1,model1))
    False
   model2 = {A:False, B:True}
    print(pl_true(atleast1,model2))
    True
   model3 = {A:True, B:True}
   print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    return disjoin(literals)
    "*** END YOUR CODE HERE ***"


def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"

    """
    at most one is the same as saying,
    
    for the list (a,b,c)
    
    (a,b),(b,c),(a,c) (combinations of a,b,c)
    
    for each in that list, negate each item
    
    (~a OR ~b) AND (~b OR ~c) AND (~a OR ~c)
    
    if a and b are both true, then that means first phrase fails -> all fails
    
    if a, and b, and c are all true same happens
    
    if a, b, c all false -> all are true and passes
    
    
    """

    combs = itertools.combinations(literals, 2)
    listNegatedTuples = []
    for tuple in combs:
        negtuple = [~p for p in tuple]  # negate the tuple's elements and

        negatedDisjoinedPhrase = disjoin(
            *negtuple)  # make a new Phrase that is the negated elements of the original tuple

        listNegatedTuples.append(negatedDisjoinedPhrase)  # makes a list of the negated phrases

    return conjoin(*listNegatedTuples)  # conjoin each element of listNegatedtuples (* operator)

    "*** END YOUR CODE HERE ***"


def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """

    """
    this should be easy because of the atMostOne function
    
    i just need to take care of the case where 0 are true. so i should append (a or b or c or d ... ) to atmostone 
    """
    "*** BEGIN YOUR CODE HERE ***"
    atMostOnePhrase = atMostOne(literals)

    disjLiterals = disjoin(*literals)

    return conjoin(atMostOnePhrase, disjLiterals)

    "*** END YOUR CODE HERE ***"


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, time = parsed
            plan[int(time)] = action
    # return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    possibilities = []
    if not walls_grid[x][y + 1]:
        possibilities.append(PropSymbolExpr(var_str, x, y + 1, t - 1)
                             & PropSymbolExpr('South', t - 1))
    if not walls_grid[x][y - 1]:
        possibilities.append(PropSymbolExpr(var_str, x, y - 1, t - 1)
                             & PropSymbolExpr('North', t - 1))
    if not walls_grid[x + 1][y]:
        possibilities.append(PropSymbolExpr(var_str, x + 1, y, t - 1)
                             & PropSymbolExpr('West', t - 1))
    if not walls_grid[x - 1][y]:
        possibilities.append(PropSymbolExpr(var_str, x - 1, y, t - 1)
                             & PropSymbolExpr('East', t - 1))

    if not possibilities:
        return None

    return PropSymbolExpr(var_str, x, y, t) % disjoin(possibilities)


def pacmanSLAMSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    moved_tm1_possibilities = []
    if not walls_grid[x][y + 1]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x, y + 1, t - 1)
                                       & PropSymbolExpr('South', t - 1))
    if not walls_grid[x][y - 1]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x, y - 1, t - 1)
                                       & PropSymbolExpr('North', t - 1))
    if not walls_grid[x + 1][y]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x + 1, y, t - 1)
                                       & PropSymbolExpr('West', t - 1))
    if not walls_grid[x - 1][y]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x - 1, y, t - 1)
                                       & PropSymbolExpr('East', t - 1))

    if not moved_tm1_possibilities:
        return None

    moved_tm1_sent = conjoin(
        [~PropSymbolExpr(var_str, x, y, t - 1), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_tm1_possibilities)])

    unmoved_tm1_possibilities_aux_exprs = []  # merged variables
    aux_expr_defs = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, t - 1)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, t - 1)
        unmoved_tm1_possibilities_aux_exprs.append(wall_dir_combined_literal)
        aux_expr_defs.append(wall_dir_combined_literal % wall_dir_clause)

    unmoved_tm1_sent = conjoin([
        PropSymbolExpr(var_str, x, y, t - 1),
        disjoin(unmoved_tm1_possibilities_aux_exprs)])

    return conjoin([PropSymbolExpr(var_str, x, y, t) % disjoin([moved_tm1_sent, unmoved_tm1_sent])] + aux_expr_defs)


def pacphysics_axioms(t, all_coords, non_outer_wall_coords):
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"

    for (x, y) in all_coords:
        pacphysics_sentences.append(PropSymbolExpr(wall_str, x, y) >> ~ PropSymbolExpr(pacman_str, x, y, t))

    allPossiblePositions = []
    for (x, y) in non_outer_wall_coords:
        allPossiblePositions.append(PropSymbolExpr(pacman_str, x, y, t))

    onePossibleLocation = exactlyOne(allPossiblePositions)
    pacphysics_sentences.append(onePossibleLocation)

    allPossibleDirections = []
    for direction in DIRECTIONS:
        allPossibleDirections.append(PropSymbolExpr(direction, t))

    onePossibleDirection = exactlyOne(allPossibleDirections)
    pacphysics_sentences.append(onePossibleDirection)

    # print((pacphysics_sentences))

    return conjoin(pacphysics_sentences)


def check_location_satisfiability(x1_y1, x0_y0, action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - problem = An instance of logicAgents.LocMapProblem
    Return:
        - a model proving whether Pacman is at (x1, y1) at time t = 1
        - a model proving whether Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    loc_0 = PropSymbolExpr(pacman_str, x0, y0, 0)
    axioms_0 = pacphysics_axioms(0, all_coords, non_outer_wall_coords)
    action_0 = PropSymbolExpr(action0, 0)
    successoraxioms_0 = allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords)
    axioms_1 = pacphysics_axioms(1, all_coords, non_outer_wall_coords)
    action_1 = PropSymbolExpr(action1, 1)

    KB.append(PropSymbolExpr(pacman_str, x0, y0, 0))
    KB.append(pacphysics_axioms(0, all_coords, non_outer_wall_coords))
    KB.append(PropSymbolExpr(action0, 0))
    KB.append(allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords))
    KB.append(pacphysics_axioms(1, all_coords, non_outer_wall_coords))
    KB.append(PropSymbolExpr(action1, 1))

    # print(KB)
    KBconj = conjoin(*KB)

    model1sentence = conjoin(KBconj, ~PropSymbolExpr(pacman_str, x1, y1, 1))
    model2sentence = conjoin(KBconj, PropSymbolExpr(pacman_str, x1, y1, 1))

    model1 = findModel(model1sentence)
    model2 = findModel(model2sentence)

    models = (model1, model2)
    return models
    "*** END YOUR CODE HERE ***"


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2),
                                        range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    KB = []

    "*** BEGIN YOUR CODE HERE ***"

    initial_loc_Expr = PropSymbolExpr(pacman_str, x0, y0, 0)
    KB.append(initial_loc_Expr)

    successors = []
    for t in range(50):
        print(f"t = {t}")

        list_expr_coords_t = []
        for (x, y) in non_wall_coords:  # one loc at timestep t
            list_expr_coords_t.append(PropSymbolExpr(pacman_str, x, y, t))

        exactOneLocT = exactlyOne(list_expr_coords_t)
        KB.append(exactOneLocT)

        if t != 0:  # it terminates if successors are found at zero i think
            for (x, y) in non_wall_coords:
                successors.append(pacmanSuccessorStateAxioms(x, y, t, walls_grid=problem.walls))

            print("FOUND SUCCESSORS")
            if successors != None and len(successors) != 0:
                succesors_Expr = conjoin(*successors)

            KB.append(succesors_Expr)

            goal_coords = problem.goal
            a, b = goal_coords
            goal_assert_expr_t = PropSymbolExpr(pacman_str, a, b, t)


            possibleActions = []
            # exactly one action
            for action in actions:
                if t != 0:
                    possibleActions.append(PropSymbolExpr(action, t - 1))

            oneActionExpr = conjoin(exactlyOne(possibleActions))
            KB.append(oneActionExpr)

            KBconj = conjoin(KB)
            model = findModel(conjoin(KBconj,goal_assert_expr_t))

        if t == 0:
            goal_coord_0 = problem.goal
            g, h = goal_coord_0
            goal_0_Expr = PropSymbolExpr(pacman_str, g, h, 0)
            goal_successor = pacmanSuccessorStateAxioms(g, h, 0, walls_grid=problem.walls)
            goal_full_Expr = conjoin(goal_0_Expr, goal_successor)
            model = findModel(conjoin(initial_loc_Expr, exactOneLocT, goal_full_Expr))
            print("TIME ZERO MODEL")

        if model:
            actionSeq = extractActionSequence(model, actions)
            return actionSeq

    "*** END YOUR CODE HERE ***"


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()
    print(food)
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    # locations = list(filter(lambda loc : loc not in walls_list, all_coords))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    initial_loc_Expr = PropSymbolExpr(pacman_str, x0, y0, 0)
    KB.append(initial_loc_Expr)

    successors = []
    for t in range(50):
        print(f"t = {t}")

        list_expr_coords_t = []
        for (x, y) in non_wall_coords:  # one loc at timestep t
            list_expr_coords_t.append(PropSymbolExpr(pacman_str, x, y, t))

        exactOneLocT = exactlyOne(list_expr_coords_t)
        KB.append(exactOneLocT)

        if t != 0:  #case where time is NOT zero ie 1-49

            for (x, y) in non_wall_coords:
                successors.append(pacmanSuccessorStateAxioms(x, y, t, walls_grid=problem.walls))

            print("FOUND SUCCESSORS")
            if successors != None and len(successors) != 0:
                succesors_Expr = conjoin(*successors)

            KB.append(succesors_Expr)





            possibleActions = []
            # exactly one action
            for action in actions:
                if t != 0:
                    possibleActions.append(PropSymbolExpr(action, t - 1))

            oneActionExpr = conjoin(exactlyOne(possibleActions))
            KB.append(oneActionExpr)

            KBconj = conjoin(KB)

            consumed_food = []
            for (x, y) in food:
                assert_pacman_at_food_expr = []
                for up_to_t in range(0, t):
                    assert_pacman_at_food_expr.append(PropSymbolExpr(pacman_str, x, y, up_to_t))
                assert_pacman_at_food_expr_disj = disjoin(assert_pacman_at_food_expr)
                consumed_food.append(assert_pacman_at_food_expr_disj)
            assert_consumed_food_t = conjoin(consumed_food)
            assert_sentence = conjoin(KBconj, assert_consumed_food_t)
            model = findModel(assert_sentence)

        if t == 0: #case where time is 0

            consumed_food = []
            for (x,y) in food:
                consumed_food.append(PropSymbolExpr(pacman_str, x, y, t))
            assert_consumed_food_0 = conjoin(consumed_food)
            assert_sentence = conjoin(initial_loc_Expr, exactOneLocT, assert_consumed_food_0)
            model = findModel(assert_sentence)



            print("TIME ZERO MODEL")

        if model:
            actionSeq = extractActionSequence(model, actions)
            return actionSeq
    "*** END YOUR CODE HERE ***"


# Helpful Debug Method
def visualize_coords(coords_list, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


def sensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                    PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t, percepts):
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], t)
        percept_unit_clauses.append(percept_unit_clause)  # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t, percepts):
    """
    SLAM uses a weaker num_adj_walls sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    num_adj_walls = sum(percepts)
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(
                combo_var % (PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSLAMSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def localization(problem, agent):
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    debug = False

    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    possible_locs_by_timestep = []
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"
    return possible_locs_by_timestep


def mapping(problem, agent):
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]
    known_map_by_timestep = []

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"
    return known_map_by_timestep


def slam(problem, agent):
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]
    known_map_by_timestep = []
    possible_locs_by_timestep = []

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"
    return known_map_by_timestep, possible_locs_by_timestep


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

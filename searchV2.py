"""Search (Chapters 3-4)
The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""

from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, PriorityQueue, name,
    distance, vector_add
)

#deque is a doubly ended queue that allows for O(1) append/pop operations
#Ex: to declare a deque => deque1 = deque(['name','age','DOB'])
from collections import defaultdict, deque
import math
import random
import sys
import bisect
from operator import itemgetter


infinity = float('inf')

# ______________________________________________________________________________


class Problem(object):

    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        #on first call, state == 'Arad'
        #isinstance returns true or false
        #this is the case where node.state == self.goal
        if isinstance(self.goal, list):
            print("self.goal is of type list")
            #this just returns a T/F value
            #and tells the bfs algo to return the solution node
            return is_in(state, self.goal)
        #this is the case where the current node isnt a goal.
        else:
            ############DEBUG MESSAGE##########################
            """ print("This node is not a goal node") """
            ############DEBUG MESSAGE##########################
            """ if(state == 'Bucharest'):
                print("Nevermind, this is bucharest node") """
            #returns false for all nodes, except for BUcharest (solution node)
            #this is why bfs algo if statement is not executed
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

# ______________________________________________________________________________
#Our problem subclass:
class MyProblem(Problem):
    """ Searching from a node to another in a given graph """
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """ The nodes that are reachable from a certain state """
        #Testing what actions returns: It returns a cities inner keys
        ############DEBUG MESSAGE##########################
        """ if A == 'Arad':
            print("Actions method called: Nodes reachable from Arad", list(self.graph.get(A).keys())) """

        #A is self.state from node.expand(), Ex: self.state == 'Arad'
        return list(self.graph.get(A).keys())
        
    def result(self, state, action):
        """ The result of an action(walking to a neighbour),
        is just that neighbour """
        return action

    #when this method is called in def child_node(), self.path_cost represents cost_so_far, which in Arad's case, is 0
    #A==self.state('Arad'), B==next_state('Zerind'), 
    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)

    def find_min_edge(self):
        m = infinity
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)
        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node.state], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return 10000



class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        #state is a pos arg, and is equal to problem.initial = 'Arad'
        self.state = state
        ############DEBUG MESSAGE##########################
        """ print(self.state, "has been passed to init") """
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    #According to the official documentation, __repr__ is used to compute the “official” string representation of an object and is typically used for debugging.
    #My hunch: This special method is called when an instance of this Node class is output as a string. So for example, the bfs method returns 
    #a solution node (bfs). When bfs is passed through the print() function, this __repr__ method is triggered
    #Original: <Node {}>
    def __repr__(self):
        return "{}:".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        #Ex:Takes all Arad's reachable cities (Ex:Zerind) and passes it to child_node()
        #Will return a LIST of nodes connected to 'problem' node passed in
        return [self.child_node(problem, action)
                #Ex:self.state == 'Arad'
                #actions returns a list of a state's reachable cities(nodes)
                for action in problem.actions(self.state)]

    #initializes a city's child nodes
    def child_node(self, problem, action):
        """[Figure 3.10]"""
        #returns the next city(Ex:if state=='Arad' returns 'Zerind' b/c thats the next element returned from action list)
        #So now next_state == 'Zerind'
        next_state = problem.result(self.state, action)

        #next_state param can be named anything, it will represent the child node's self.state when initialized in Node() class
        #next_node is a new instance of Node() class, and represents a connected city to 'Arad', so for ex next_node = Zerind
        #This is where the costs and distances are figured out
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        #I think this is close to displaying each node's path cost
        """ print("City, and distance traveled so far:", next_node, next_node.path_cost) """
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________


class SimpleProblemSolvingAgentProgram:

    """Abstract framework for a problem-solving agent. [Figure 3.1]"""

    def __init__(self, initial_state=None):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError

# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """Search the shallowest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_graph_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Does not get trapped by loops.
        If two paths reach a state, only use the first one. [Figure 3.7]"""
    frontier = [(Node(problem.initial))]  # Stack
    explored = set()

    exp = 0
    while frontier:
        node = frontier.pop()
        exp = exp + 1

        if problem.goal_test(node.state):
            ##########DEBUG MESSAGE###############
            """ print("Number of nodes visited: ", exp) """
            return node
        
        #This line prints the current node's path cost from the root
        print("Current city, and distance traveled so far:", node.state, node.path_cost)
        explored.add(node.state)
        #extend is a stack method, just appends elements to the rightmost part of deque/stack
        #This block returns a list of child nodes connected to current node, and stores them in the stack
        #'child for child' is a list comprehension
        frontier.extend(child for child in node.expand(problem)
                        #once you have a list of connected child nodes, check to see if you've already seen them.
                        #if not, theyll be added to the frontier using extend()
                        if child.state not in explored and
                        child not in frontier)
        
        #Previous attempt at printing out distances
        """ for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                print(child.state, ":", child.path_cost) """
        
    return None


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    ###Ex: problem.initial = 'Arad', its passed into Node class

    node = Node(problem.initial)
    #node.state is problem.initial == 'Arad'
    #we had to convert pro.init to node.state to use prob.goal_test()

    #this if should only execute when Bucharest node is found
    if problem.goal_test(node.state):
        ############DEBUG MESSAGE##########################
        """ print("found goal") """
        return node

    #frontier are the nodes waiting in a queue to be explored
    #passing a single element list(our current node) into the deque called frontier
    frontier = deque([node])
    #set() is like a list but no multiple occurences of the same element
    ##you init a set with a list
    explored = set()

    exp = 0
    #while frontier has a value, and not empty:
    while frontier:
        #overwrites current node with node from left end of the deque,
        #for root node(Arad), node just becomes Arad again
        node = frontier.popleft()
        exp = exp + 1

        #__repr__ is called here
        #node that has been popped is printed out
        """ print(node) """
        #once we've visited a node, we add it to explored set. we add the 'state' of the node, or the name of the city ex 'Arad'
        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                #checks to see if current node is solution node
                if problem.goal_test(child.state):
                    #returns solution node(Bucharest)
                    return child
                #To print out the traversed nodes and the distance from the root, I added this line 
                print("City and its' distance from root city: ", child.state, ":", child.path_cost)
                frontier.append(child)
    print("after bfs while")
    return None


def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    exp = 0
    while frontier:
        node = frontier.pop()
        exp = exp + 1

        #print(node)
        if problem.goal_test(node.state):
            return node

        print("Current city, and distance traveled so far:", node.state, node.path_cost)
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

# _____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.

# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:

    """A graph connects nodes (vertices) by edges (links).  Each edge can also
    have a length associated with it.  The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C.  You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added.  You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B.  'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict = graph_dict, directed=False)



""" [Figure 3.2]
Simplified road map of Romania
"""
romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))



class GraphProblem(Problem):

    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = infinity
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity



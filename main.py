from search import *

class MyProblem(Problem):
    """ Searching from a node to another in a given graph """
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """ The nodes that are reachable from a certain state """
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """ The result of an action(walking to a neighbour),
        is just that neighbour """
        return action

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




def main():

    # Task 1: Romania problem, from Arad to Bucharest

    answer = input("Would you like to run a searching algorithm? Answer Y or N:")

    while(answer == 'Y'):

        algo = input("Choose which algo to run: B/D/A* ")

        start = input("Please enter starting city:")
        end = input("Plase enter destination(Bucharest):")
        

        

        if algo == 'B':
            romania_problem = MyProblem(start, end, romania_map)

            print("--- Breadth First Search ---")
            print("BFS route:")
            bfs = breadth_first_graph_search(romania_problem)
            print("Itinerary:")
            print(bfs.solution())
            #this was a simple fix, just removed the () from path.cost() and it prints the total distance
            print("Total Distance: {0}".format(bfs.path_cost))
        elif algo == 'D':

            romania_problem = MyProblem(start, end, romania_map)
            print("\n--- Depth First Search ---")
            print("DFS Route:")
            dfs = depth_first_graph_search(romania_problem)
            print("Itinerary:")
            print(dfs.solution())
            print("Total Distance: {0}".format(dfs.path_cost))
        elif algo == 'A*':
            
            romania_problem = MyProblem(start, end, romania_map)
            print("\n--- A* Search: ---")
            print("A* Route:")
            astar = astar_search(romania_problem)
            print("Itinerary:")
            print(astar.solution())
            #shit does not work
            """ print("Path cost: {0}".format(astar_search.path_cost)) """
        else:
            pass
        answer = input("Would you like to run another algorithm? Enter Y or N:")
        if(answer == 'N'):
            print("Goodbye")
        

            


if __name__ == '__main__':
    main()

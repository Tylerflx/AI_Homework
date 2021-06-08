from searchV2 import *


def main():

    # Task 1: Romania problem, from Arad to Bucharest

    answer = input("Would you like to run a search algorithm? Answer Y or N:")

    while(answer == 'Y' or answer == 'y'):

        algo = input("Choose which algo to run: B/D/A* ")
        start = input("Please enter starting city:")
        """ end = input("Plase enter destination(Bucharest):") """
        end = "Bucharest"
        print("Running search algorithm for Bucharest...\n")

        if algo == 'B':
           
            #initializes problem to be processed
            romania_problem = MyProblem(start, end, romania_map)

            print("--- Breadth First Search ---")
            print("BFS route:")

            #returns a solution node
            bfs = breadth_first_graph_search(romania_problem)
            print("Itinerary: ", bfs.solution())
            #this was a simple fix, just removed the () from path.cost() and it prints the total distance
            print("Total Distance to Bucharest: {0}\n".format(bfs.path_cost))

        elif algo == 'D':

            romania_problem = MyProblem(start, end, romania_map)
            print("\n--- Depth First Search ---")
            print("DFS Route:")
            dfs = depth_first_graph_search(romania_problem)

            print("Itinerary: ", dfs.solution())
            print("Total Distance to Bucharest: {0}\n".format(dfs.path_cost))

        elif algo == 'A*':
            
            romania_problem = MyProblem(start, end, romania_map)
            print("\n--- A* Search: ---")
            print("A* Route:")
            astar = astar_search(romania_problem)
            print("Itinerary: ", astar.solution())
            print("Total Distance to Bucharest: {0}\n".format(astar.path_cost))

        else:
            pass
        
        answer = input("Would you like to run another algorithm? Enter Y or N:")
        if(answer == 'N'):
            print("Goodbye")
        

            


if __name__ == '__main__':
    main()

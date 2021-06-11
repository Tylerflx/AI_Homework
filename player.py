import math
import random

class Player:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass

#human player class
class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    #get move from input
    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            #this will check if the user's input is valid
            try:
                val = int(square)
                if val not in game.available_moves(): #input should be available white empty spaces
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.') #raise error if the user put wrong inputs
        return val

class AIPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
    def get_move(self, game):
        if len(game.available_moves()) == 9: #if the game just started
            square = random.choice(game.available_moves()) #make a random choice
        else:
            square = self.minmax(game, self.letter)['position'] #other wise started minmax algorithm to play the game
        return square
    
    #minmax algorithm
    def minmax(self, state, player):
        max_player = self.letter
        other_player = 'O' if player == 'X' else 'X'

        #create a tree of decision to make the best move
        if state.current_winner == other_player:
            return {'position': None,
                    'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1
                            * (state.num_empty_squares() + 1)
                    }
        elif not state.empty_squares():
            return {'position': None, 'score': 0}
        if player == max_player:
            best = {'position': None, 'score': -math.inf}
        else:
            best = {'position': None, 'score': math.inf}
        
        #once calculated the best move, start make a move
        for posible_move in state.available_moves():
            #step 1
            state.make_move(posible_move, player)
            #step 2
            sim_score = self.minmax(state, other_player)
            #step 3
            state.board[posible_move] = ' '
            state.current_winner = None
            sim_score['position'] = posible_move

            #step 4
            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best

import math
import random

class Player:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass


# class ComputerPlayer(Player):
#     def __init__(self, letter):
#         super().__init__(letter)

#     def get_move(self, game):
#         square = random.choice(game.available_moves())
#         return square

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            #this will check if the user's input is valid
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val

class AIPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minmax(game, self.letter)['position']
        return square
    
    def minmax(self, state, player):
        max_player = self.letter
        other_player = 'O' if player == 'X' else 'X'

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

import perception # perception class

class HumanPlayer():

    def __init__(self, name="HumanPlayer"):
        self.name = name

    """Player that chooses a move according to user's input."""
    def move(self, game, legal_moves, time_left):
        choice = {}

        if not len(legal_moves):
            print ("No more moves left.")
            return None, None

        # look for board state change with camera

        # if board state change -> get player move -> see if move is valid -> play move or print illegal move

        while not valid_choice:
            try:
                index = int(input('Select move index [1-'+str(len(legal_moves))+']:'))
                valid_choice = 1 <= index <= len(legal_moves)

                if not valid_choice:
                    print('Illegal move of queen! Try again.')
            except Exception:
                print('Invalid entry! Try again.')

        return choice[index]

    def get_name(self):
        return self.name
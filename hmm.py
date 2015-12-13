import random
from collections import Counter


class HMM:
    """
    A simple hidden Markov model.
    The probability matrix is a square matrix represented this way:
    ```
          +-----+-----+-----+
          |  A  |  B  |  C  |
    +-----+-----+-----+-----+
    |  A  |  a  |  b  |  c  |
    +-----+-----+-----+-----+
    |  B  |  d  |  e  |  f  |
    +-----+-----+-----+-----+
    |  C  |  i  |  j  |  k  |
    +-----+-----+-----+-----+
    ```
    with:
     - `a` the probability for the state A to got to state A
     - `b` the probability for the state A to got to state B
     - `c` the probability for the state A to got to state C
     - ...
    Instead of using a 2D array, we use a dictionary of counters.
    The dictionary contains the rows indexed by each state, each row contains counters indexed again by each state.
    Using dictionary is usually simpler (we do not have to handle hash the elements), and faster than using an array
    (O(1) instead of O(n) to access it, operation we use a lot).
    Using a 2D array + a separate index + a hash function would be a bit faster, and a lot less memory consuming,
    but more confusing and less generic.
    """

    def __init__(self, states):
        """
        Create a hidden markov chain
        :param states: a set of all the different states
        """
        self.states = states
        # We create the matrix
        self.matrix = {state: Counter() for state in self.states}

    def next_state(self, current_state):
        """
        Generate a next state according to the matrix's probabilities
        :param current_state: the state to start with
        :return: a next state
        """
        row = self.matrix[current_state]  # We get the row associated with the current state

        # Here, we want to get an random element in respect to the probabilities in the row. We do this in O(n) by
        # selecting a random number between 0 and 1, walking though the elements and their probability in the list,
        # subtracting the probabilities from our number until it is 0 or less.
        # But since the probabilities in the row do not add up to 1 (it is only a part of the matrix), we generate a
        # number between 0 and the sum of probabilities in the row
        total = sum(row.values())
        number = random.uniform(0.0, total)  # Generate a number in [0, total] with equal probability
        for state, probability in row.items():
            number -= probability
            if number <= 0:
                return state

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

    def probability_of_chain(self, chain):
        """
        Compute the probability for a given chain of text to occur.
        :param chain: the chain of states as an ordered list
        :return: the probability for it to happen
        """
        # If the chain is empty, we return a null probability
        if len(chain) == 0:
            return 0

        # If the chain is made of a single state, we return 1 if the state exists, 0 otherwise
        if len(chain) == 1:
            if chain[0] in self.matrix:
                return 1
            else:
                return 0

        probability = 1.0
        for state, next_state in zip(chain, chain[1:]):
            row = self.matrix[state]  # The row associated with the state

            # If the transition between state and next_state is impossible, the probability of the chain is 0
            if next_state not in row:
                return 0

            probability *= row[next_state]
        return probability

    def generate_chain(self, start_state, size):
        """
        Generate of probable chain of state, respecting the probabilities in the matrix
        :param start_state: the starting state of the chain
        :param size: the size of the chain
        :return: the chain as an ordered list
        """
        chain = [start_state]
        state = start_state
        for n in range(0, size):
            state = self.next_state(state)
            chain.append(state)
        return chain

    def train(self, chain):
        """
        Train the model on an example chain
        :param chain: the chain of state as an ordered list
        """
        # We read the text two words by two words
        for s1, s2 in zip(chain, chain[1:]):
            self.matrix[s1][s2] += 1

        # We normalize the matrix, transforming occurrences into probabilities
        factor = 1.0 / (len(chain) - 1)  # Instead of dividing by the number of words - 1, we use a multiplication
        for row in self.matrix.values():
            for state, occurences in row.items():
                row[state] *= factor

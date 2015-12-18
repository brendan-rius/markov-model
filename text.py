import re
from hmm import HMM


class TextHMM(HMM):
    """
    A HMM that can be trained with a text and that is able to generate sentences from it.
    Here the states are the words in the vocabulary of the text.
    """

    def __init__(self, text):
        self.text = text
        # We splt the text into words
        self.words = self._lex(self.text)
        # The vocabulary is the set of different states
        self.states = set(self.words)
        super().__init__(self.states)

    def train(self):
        """
        Train the HMM using a text
        """
        # We read the text two words by two words
        for w1, w2 in zip(self.words, self.words[1:]):
            self.matrix[w1][w2] += 1

        # We normalize the matrix, transforming occurrences into probabilities
        factor = 1.0 / (len(self.words) - 1)  # Instead of dividing by the number of words - 1, we use a multiplication
        for row in self.matrix.values():
            for state, occurences in row.items():
                row[state] *= factor

    def _lex(self, text):
        """
        Splits the text into words
        :param text: the text
        :return: a list of words
        """
        # Split at each character or sequence of character that is not a valid word character (in the \w regex class)
        return re.compile('[^\w]+').split(text)

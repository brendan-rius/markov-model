import re
from hmm import HMM


class TextHMM(HMM):
    """
    A HMM that can be trained with a text and that is able to generate sentences from it.
    Here the states are the words in the vocabulary of the text.
    """

    def __init__(self, text):
        # We split the text into words
        self.words = self._lex(text)
        # The vocabulary is the set of different states
        self.states = set(self.words)
        super().__init__(self.states)

    def train(self):
        super().train(self.words)

    def _lex(self, text):
        """
        Splits the text into words
        :param text: the text
        :return: a list of words
        """
        # Split at each character or sequence of character that is not a valid word character (in the \w regex class)
        return re.compile('[^\w]+').split(text)

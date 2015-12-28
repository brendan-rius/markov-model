# A stupid Markov Model :)

Markov model playground.
Currently only handles discrete-time, discrete-space, first order Markov chains.

## Requirements

+ Python 3

## General markov model

The class `MarkovModel` represents a general first-order, space-discrete and time-discrete markov model.

It has to be initialized with its space (a set of all possible states) like:

```python
model = MarkovModel({'A', 'B', 'C'}) // we create a model with 3 different states
```

It then has to be trained with a markov chain of those states:
```python
model.train(['A', 'A', 'A', 'B', 'A', 'C', 'C', 'A', 'A', 'A', 'A', 'A']) // we train it against this ordered sequence of states
```

And it can then predict a the next state of a state:
```python
model.next_state('A') // will likely return A, since most of the times, A are followed by A during the training phase
```

Or predict the probability of a sequence:
```python
model.probability_of_chain(['A', 'C', 'C']) // the probability for this sequence to be generated
```

It can even create a sequence from a starting state:
```python
model.generate_chain('C', 5) // generate a 5-states sequence beginning with 'C', using the probability of the dataset it has be trained with
```

## Available Markov Models implementations

### Text Markov Model

The time is the order of appearance of tokens (words) in a text, the space is made of the set of words.

Can generate a sentence after having been trained with a text. Code example:
The sentences will look like english sentences, but will have no meaning.

```python
from text import TextHMM

with open('text.txt', 'r') as file:
    text = file.read()
hmm = TextHMM(text)  # We create the HMM from the text
hmm.train()  # We train it
print(' '.join(hmm.generate_chain("the", 7)))  # We generate 7 words, starting with "the"

```

```
the very good sort of injustice towards Highbury
```

The example test `text.txt` comes from Shakespeare.

## Improvements to be made

+ Handle not only first order Markov Model but n-th order chains
+ Live training (being able to use the model as we train it, that is being to update the probability matrix after having
 created it)

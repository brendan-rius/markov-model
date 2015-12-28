# A stupid Markov Model :)

Markov model playground

## Requirements

+ Python 3

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

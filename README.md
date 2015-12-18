# A stupid Hidden Markov Model

A stupid example of a hidden Markov Model.

## Requirements

+ Python 3

## Available HMM

### Text HMM

Can generate a sentence after having been trained with a text. Code example:
The sentences will look like english sentences, but will have no meaning.

```
from text import TextHMM

with open('text.txt', 'r') as file:
    text = file.read()
hmm = TextHMM(text)  # We create the HMM from the text
hmm.train()  # We train it
print(' '.join(hmm.generate_sentence("the", 7)))  # We generate 7 words, starting with "the"

```

```
the very good sort of injustice towards Highbury
```

The example test `text.txt` comes from Shakespeare.
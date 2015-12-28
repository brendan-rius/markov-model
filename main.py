from text import TextMarkovModel

with open('text.txt', 'r') as file:
    text = file.read()
hmm = TextMarkovModel(text)
hmm.train()
print(' '.join(hmm.generate_chain("the", 7)))

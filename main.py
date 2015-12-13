from text import TextHMM

with open('text.txt', 'r') as file:
    text = file.read()
hmm = TextHMM(text)
hmm.train()
print(' '.join(hmm.generate_sentence("the", 7)))

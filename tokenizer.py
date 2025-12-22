class tokenization: #initializing tokanizer
   def __init__(self):
      self.punctuations = "!@#$%^&*(),./;'[]\\|?"                       # added stopwords and punctuations because it adds only noise
      self.stopwords = [
            "i", "am", "is", "are", "the", "a", "an",
            "and", "or", "of", "to", "in", "on", "for", "with"
          ]
      self.tokenized_text = []
      self.vocab = {}
      self.index = 0
   def tokenizer(self,text):
       text = text.lower()
       cleared_text = ""
       for char in text:
          if char not in self.punctuations:                                  # text cleaning
             cleared_text += char
       cleared_text = cleared_text.split()
       tokens = []
       for token in cleared_text:
          if token not in self.stopwords:
             tokens.append(token)
       return tokens
   def make_vocab(self,tokens):
       for words in tokens:
         if words not in self.vocab:
            self.vocab[words] = self.index                                      # adding to Vocab
            self.index += 1
       return self.vocab
    
nlp = tokenization()
text = input("How can I help U Darling : ")
a = nlp.tokenizer(text)
print(a)
b = nlp.make_vocab(a)
print(b)
import math

class TfidfNaiveBayes:

    def __init__(self):
        self.punctuations = "!@#$%^&*(),./;'[]\\|?\"’“”"
        self.stopwords = [
            "i", "am", "is", "are", "the", "a", "an",
            "and", "or", "of", "to", "in", "on", "for", "with"
        ]

        self.vocab = {}
        self.index = 0
        self.doc_count = 0
        self.word_doc_freq = {}


        self.class_doc_count = {}
        self.class_tfidf_sum = {}
        self.total_docs = 0

    def tokenize(self, text):
        text = text.lower()
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        cleaned = ""

        for char in text:
            if char not in self.punctuations:
                cleaned += char

        words = cleaned.split()

        tokens = []
        for word in words:
            if word not in self.stopwords:
                tokens.append(word)

        return tokens
    

    def update_vocab_df(self, tokens):
          self.doc_count += 1

          for word in set(tokens):
              self.word_doc_freq[word] = self.word_doc_freq.get(word, 0) + 1

          for word in tokens:
              if word not in self.vocab:
                  self.vocab[word] = self.index
                  self.index += 1

    def tfidf_vector(self, tokens):
        tf = [0] * len(self.vocab)
        total = len(tokens)

        for word in tokens:
            if word in self.vocab:
              idx = self.vocab[word]
              tf[idx] += 1

        for i in range(len(tf)):
            tf[i] /= total

        tfidf = [0] * len(self.vocab)

        for word, idx in self.vocab.items():
            idf = math.log(self.doc_count / self.word_doc_freq[word])
            tfidf[idx] = tf[idx] * idf

        return tfidf

    def train(self, text, label):
      tokens = self.tokenize(text)

      old_vocab_size = len(self.vocab)

      self.update_vocab_df(tokens)

      new_vocab_size = len(self.vocab)

      if new_vocab_size > old_vocab_size:
          for cls in self.class_tfidf_sum:
              extra = new_vocab_size - old_vocab_size
              self.class_tfidf_sum[cls].extend([0] * extra)

      tfidf = self.tfidf_vector(tokens)

      self.total_docs += 1
      self.class_doc_count[label] = self.class_doc_count.get(label, 0) + 1

      if label not in self.class_tfidf_sum:
          self.class_tfidf_sum[label] = [0] * new_vocab_size

    
      for i in range(len(tfidf)):
          self.class_tfidf_sum[label][i] += tfidf[i]


    def predict(self, text):
        tokens = self.tokenize(text)
        tokens = [w for w in tokens if w in self.vocab]

        tfidf = self.tfidf_vector(tokens)

        scores = {}

        for label in self.class_doc_count:
        
            score = math.log(self.class_doc_count[label] / self.total_docs)

            total_weight = sum(self.class_tfidf_sum[label]) + 1e-9

            for i in range(len(tfidf)):
                prob = (self.class_tfidf_sum[label][i] + 1) / total_weight
                score += tfidf[i] * math.log(prob)

            scores[label] = score

        return max(scores, key=scores.get)



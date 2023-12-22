from collections import defaultdict, Counter

class NgramModel:
    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(Counter)

    def build_model(self, data):
        for line in data:
            padded_line = '#' * (self.n - 1) + line.strip()
            for i in range(len(padded_line) - self.n):
                ngram = padded_line[i:i + self.n - 1]
                next_char = padded_line[i + self.n - 1]
                self.model[ngram][next_char] += 1

    def predict_next_chars(self, sequence, num_predictions=4):
        sequence = sequence.lower()
        if len(sequence) < self.n - 1:
            sequence = '#' * (self.n - 1 - len(sequence)) + sequence
        
        ngram_key = sequence[-(self.n - 1):]
        predictions = self.model[ngram_key]
        
        return predictions.most_common(num_predictions)

    def predict_next_chars_no_space(self, sequence, num_predictions=4):
        sequence = sequence.lower()
        if len(sequence) < self.n - 1:
            sequence = '#' * (self.n - 1 - len(sequence)) + sequence
        
        ngram_key = sequence[-(self.n - 1):]
        predictions = self.model[ngram_key]
        
        filtered_predictions = [pred for pred in predictions.items() if pred[0] != ' ']
        return sorted(filtered_predictions, key=lambda x: x[1], reverse=True)[:num_predictions]

# Function to preprocess data
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [''.join(e for e in line.lower() if e.isalnum() or e.isspace()) for line in data]

# Initialization of the model
def init_model():
    file_path = './data/phrases2.txt'  # Assuming the file is in the 'data' directory
    processed_data = preprocess_data(file_path)
    ngram_model = NgramModel(n=2)  # Using bigrams (2-grams)
    ngram_model.build_model(processed_data)
    return ngram_model

# Global model instance
model_instance = init_model()


# In this setup:

# NgramModel is a class handling the n-gram model building and predictions.
# The init_model function initializes the model with data from phrases2.txt.
# model_instance is a global instance of NgramModel, ready to serve predictions.
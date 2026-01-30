from tensorflow import keras

# Load the saved model
model = keras.models.load_model('imdb_rnn_model.keras')

# Load word index (needed to convert text to numbers)
word_index = keras.datasets.imdb.get_word_index()

# Configuration (must match training settings)
MAX_LEN = 200

def predict_sentiment(review_text, word_index, model, max_len=200):
    """Predict sentiment of a text review"""
    # Convert text to sequence of integers
    words = review_text.lower().split()
    sequence = [word_index.get(word, 0) for word in words]
    
    # Pad sequence
    padded = keras.preprocessing.sequence.pad_sequences(
        [sequence], maxlen=max_len
    )
    
    # Make prediction
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

# Interactive loop
print("IMDB Sentiment Analyzer (type 'quit' to exit)")
review = input(">>> ")
while review.lower().strip() not in ['q', 'quit', 'exit', 'stop']:
    if len(review.strip()) == 0:
        review = input(">>> ")
        continue
    
    score = predict_sentiment(review, word_index, model, MAX_LEN)
    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"Sentiment: {sentiment} (score: {score:.4f})")
    review = input(">>> ")

print("Goodbye!")
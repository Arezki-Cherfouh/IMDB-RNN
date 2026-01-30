import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration optimized for limited resources
MAX_FEATURES = 5000  # Vocabulary size (reduced for memory efficiency)
MAX_LEN = 200  # Maximum review length (reduced for faster training)
BATCH_SIZE = 32  # Small batch size for memory efficiency
EPOCHS = 5  # Number of training epochs
EMBEDDING_DIM = 32  # Embedding dimension (smaller for CPU)
RNN_UNITS = 32  # RNN units (smaller for CPU)

print("Loading IMDB dataset...")
# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=MAX_FEATURES
)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# Pad sequences to ensure uniform length
print("Padding sequences...")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# model = keras.models.load_model('imdb_rnn_model.keras')

# Build the RNN model
print("\nBuilding RNN model...")
model = keras.Sequential([
    # Embedding layer: converts word indices to dense vectors
    layers.Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM),
    
    # Simple RNN layer with dropout for regularization
    layers.SimpleRNN(RNN_UNITS, dropout=0.2, recurrent_dropout=0.2),
    
    # Dense output layer for binary classification
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
print("This may take 5-10 minutes on your CPU...\n")

history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
print("\nGenerating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
print("Training plots saved to 'training_history.png'")

# Function to predict sentiment of new reviews
# def predict_sentiment(review_text, word_index):
#     # Convert text to sequence of integers
#     words = review_text.lower().split()
#     sequence = [word_index.get(word, 0) for word in words]
    
#     # Pad sequence
#     padded = keras.preprocessing.sequence.pad_sequences(
#         [sequence], maxlen=MAX_LEN
#     )
    
#     # Make prediction
#     prediction = model.predict(padded, verbose=0)[0][0]
#     return prediction

def predict_sentiment(review_text, word_index, max_features=MAX_FEATURES):
    # Convert text to sequence of integers
    words = review_text.lower().split()
    
    sequence = []
    for word in words:
        idx = word_index.get(word, 0)
        # Only include if within vocabulary range
        if idx < max_features:
            sequence.append(idx)
        else:
            sequence.append(0)  # Treat as unknown word
    
    # Pad sequence
    padded = keras.preprocessing.sequence.pad_sequences(
        [sequence], maxlen=MAX_LEN
    )
    
    # Make prediction
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

# Load word index for making predictions on new text
word_index = keras.datasets.imdb.get_word_index()

# Test with some example reviews
print("\n" + "="*60)
print("Testing with example reviews:")
print("="*60)

test_reviews = [
    "This movie was absolutely amazing! Great acting and plot.",
    "Terrible film. Waste of time and money. Very disappointed.",
    "It was okay, nothing special but not bad either."
]

for review in test_reviews:
    score = predict_sentiment(review, word_index)
    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment} (score: {score:.4f})")

# review = input(">>>")
# while review.lower().strip() not in ['q','quit','exit','stop']:
#     score = predict_sentiment(review, word_index)
#     sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
#     print(f"\nReview: {review}")
#     print(f"Sentiment: {sentiment} (score: {score:.4f})")
#     review = input(">>>")

review = input(">>> ")
while review.lower().strip() not in ['q', 'quit', 'exit', 'stop']:
    words = review[:200]
    if len(words) == 0:
        print("Please enter a review.")
        review = input(">>> ")
        continue
    review_text = ' '.join(words)
    score = predict_sentiment(review_text, word_index)
    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"\nSentiment: {sentiment} (score: {score:.4f})")
    review = input(">>> ")

# Save the model
print("\n" + "="*60)
model_path = 'imdb_rnn_model.keras'
model.save(model_path)
print(f"Model saved to '{model_path}'")
print("\nTo load the model later, use:")
print("model = keras.models.load_model('imdb_rnn_model.keras')")
print("="*60)
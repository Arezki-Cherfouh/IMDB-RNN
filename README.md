This project implements a basic sentiment analysis system using a Simple RNN on the IMDB movie reviews dataset. It is optimized for limited resources and CPU usage by reducing vocabulary size, sequence length, embedding dimensions, and RNN units.

The script loads and preprocesses the IMDB dataset, pads sequences, builds and trains a SimpleRNN model, evaluates it on a test set, and saves training accuracy and loss plots. After training, the model can predict sentiment for custom text entered via the terminal.

Requirements: Python, TensorFlow, NumPy, Matplotlib.

Run the script to train the model. After training, type a review to get a POSITIVE or NEGATIVE prediction. Type q, quit, or exit to stop. The trained model is saved as imdb_rnn_model.keras for later use.

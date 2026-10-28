import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset settings
vocab_size = 10000
max_length = 200

# Load IMDB movie reviews
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# Pad or truncate reviews to equal length
X_train = pad_sequences(
    X_train, maxlen=max_length,
    padding="post", truncating="post"
)
X_test = pad_sequences(
    X_test, maxlen=max_length,
    padding="post", truncating="post"
)

# Create LSTM model
model = models.Sequential([
    layers.Input(shape=(max_length,)),
    layers.Embedding(vocab_size, 64, mask_zero=True),
    layers.LSTM(64),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Predict the sentiment of a test review
probability = model.predict(X_test[:1], verbose=0)[0, 0]

print("Positive probability:", round(float(probability), 4))
print("Predicted sentiment:",
      "Positive" if probability >= 0.5 else "Negative")
print("Actual sentiment:",
      "Positive" if y_test[0] == 1 else "Negative")
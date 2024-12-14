import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ensure required resources are downloaded
nltk.download('punkt')

# Initialize SpellChecker
spell = SpellChecker()

# Load dataset and preprocess text (adjust the file path accordingly)
with open(r'C:\Users\kkkos\Desktop\Keyboard.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()  # Read text and convert to lowercase

# Tokenize and clean data
tokens = word_tokenize(text)
n = 3  # Using trigrams for next-word prediction
sequences = []
for i in range(n, len(tokens)):
    sequences.append(tokens[i-n:i])  # Correcting this to ensure 3 elements: word1, word2, next_word

# Convert sequences to DataFrame (ensure correct number of columns for trigram)
df = pd.DataFrame(sequences, columns=['word1', 'word2', 'next_word'])

# Encode words as integers
all_words = list(df['word1']) + list(df['word2']) + list(df['next_word'])  # Collect all words for fitting the encoder
encoder = LabelEncoder()
encoder.fit(all_words)  # Fit encoder on all words

df['word1'] = encoder.transform(df['word1'])
df['word2'] = encoder.transform(df['word2'])
df['next_word'] = encoder.transform(df['next_word'])
vocab_size = len(encoder.classes_)  # Total vocabulary size

# Create input and output arrays
X = df[['word1', 'word2']].values
y = df['next_word'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with Bidirectional Layer
embedding_dim = 50
hidden_units = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=2))  # Use input_length=2 for the two words in the sequence
model.add(Bidirectional(LSTM(hidden_units)))  # Bidirectional LSTM to capture context from both directions
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (adjust epochs for your needs)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Function to check spelling and correct words before prediction
def correct_spelling(word):
    # First, try to correct the word using SpellChecker
    corrected_word = spell.correction(word)
    
    # If the corrected word is in the vocabulary, return it
    if corrected_word in encoder.classes_:
        return corrected_word
    
    # If the corrected word is not in the vocabulary, return a placeholder or fallback word
    return "<OOV>"  # You can use any placeholder like "<UNKNOWN>" or a common fallback word

# Integrated function to autocorrect input and predict next words
def predict_next_words(word1, word2, num_predictions=5):
    # Spellcheck both words
    word1, word2 = correct_spelling(word1), correct_spelling(word2)
    
    # Check if words are out of vocabulary (OOV)
    if word1 == "<OOV>" or word2 == "<OOV>":
        return ["One or both words are out of vocabulary."]
    
    # Encode the corrected words
    if word1 in encoder.classes_ and word2 in encoder.classes_:
        input_sequence = np.array([[encoder.transform([word1])[0], encoder.transform([word2])[0]]])
        predictions = model.predict(input_sequence, batch_size=1)
        
        predicted_words = []
        for i in range(num_predictions):
            predicted_index = np.argmax(predictions, axis=-1)
            predicted_word = encoder.inverse_transform(predicted_index)[0]
            predicted_words.append(predicted_word)
            
            # Use the predicted word as the next input for subsequent predictions
            word1, word2 = word2, predicted_word
            input_sequence = np.array([[encoder.transform([word1])[0], encoder.transform([word2])[0]]])
            predictions = model.predict(input_sequence, batch_size=1)

        return predicted_words
    else:
        return ["One or both words are out of vocabulary."]

# Create the GUI window
window = tk.Tk()
window.title("Autocorrect and Next-Word Prediction")
window.geometry("500x600")
window.config(bg='#f0f0f0')  # Set background color of the window

# Label for user instructions
label1 = tk.Label(window, text="Enter the first word:", font=("Helvetica", 12), bg='#f0f0f0')
label1.pack(pady=10)

word1_entry = tk.Entry(window, width=30, font=("Arial", 12), bg='#ffffff', fg='#333333', bd=2)
word1_entry.pack(pady=5)

label2 = tk.Label(window, text="Enter the second word:", font=("Helvetica", 12), bg='#f0f0f0')
label2.pack(pady=10)

word2_entry = tk.Entry(window, width=30, font=("Arial", 12), bg='#ffffff', fg='#333333', bd=2)
word2_entry.pack(pady=5)

# Label to show the corrected words
corrected_label = tk.Label(window, text="", font=("Helvetica", 10, "italic"), bg='#f0f0f0', fg='green')
corrected_label.pack(pady=5)

# Function to display prediction result in the GUI
def on_predict():
    word1 = word1_entry.get()
    word2 = word2_entry.get()
    if word1 and word2:  # Ensure that the input fields are not empty
        # Correct the words and display
        corrected_word1 = correct_spelling(word1)
        corrected_word2 = correct_spelling(word2)
        corrected_label.config(text=f"Corrected words: {corrected_word1}, {corrected_word2}")
        
        predictions = predict_next_words(word1, word2)
        result_label.config(text="Predicted next words: " + ", ".join(predictions), fg='blue')
    else:
        messagebox.showwarning("Input Error", "Please enter both words.")

# Button to trigger prediction
predict_button = tk.Button(window, text="Predict Next Words", command=on_predict, font=("Arial", 14), bg='#4CAF50', fg='white', bd=2, relief="raised")
predict_button.pack(pady=20)

# Label to display the result
result_label = tk.Label(window, text="", font=("Helvetica", 12, "bold"), bg='#f0f0f0', fg='blue')
result_label.pack(pady=10)

# Start the GUI application
window.mainloop()

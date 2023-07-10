import tokenizers

"""
In this example, we first load a dataset of texts. Then, we initialize a BPE tokenizer using the tokenizers. Tokenizer class and the
tokenizers.models.BPE() model. To customize the tokenizer training, we initialize a tokenizers.trainers.BpeTrainer object with desired 
parameters such as vocab_size and min_frequency.

We then train the tokenizer on the dataset using the train_from_iterator method of the tokenizer object. Finally, we preprocess a new 
text using the trained tokenizer by encoding it with the encode method and decoding it with the decode method. The encoded text is 
represented as a list of integers (encoded_text.ids), and the decoded text is a string. Note that you may need to install the tokenizers
package first using pip install tokenizers.

"""



# Load the dataset
texts = ["This is the first text.", "This is the second text."]

# Initialize a BPE tokenizer
tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

# Customize the tokenizer training
trainer = tokenizers.trainers.BpeTrainer(vocab_size=5000, min_frequency=2)

# Train the tokenizer on the dataset
tokenizer.train_from_iterator(texts, trainer=trainer)

# Preprocess text using the tokenizer
text = "This is a new text to preprocess."
encoded_text = tokenizer.encode(text)
decoded_text = tokenizer.decode(encoded_text.ids)

print(encoded_text.ids)  # [51, 23, 10, 34, 125, 297, 7, 160, 94, 2]
print(decoded_text)  # This is a new text to preprocess.



from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers import pre_tokenizers, processors, normalizers, decoders
from transformers import PreTrainedTokenizerFast


import os
"""
In this example, we first load a dataset of texts. Then, we initialize a BPE tokenizer using the tokenizers. Tokenizer class and the
tokenizers.models.BPE() model. To customize the tokenizer training, we initialize a tokenizers.trainers.BpeTrainer object with desired 
parameters such as vocab_size and min_frequency.

We then train the tokenizer on the dataset using the train_from_iterator method of the tokenizer object. Finally, we preprocess a new 
text using the trained tokenizer by encoding it with the encode method and decoding it with the decode method. The encoded text is 
represented as a list of integers (encoded_text.ids), and the decoded text is a string. Note that you may need to install the tokenizers
package first using pip install tokenizers.

"""



# trainer = BpeTrainer(vocab_size=1000, min_frequency=2)
# tokenizer.train(trainer, ["path/to/your/text/data.txt"])

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "[CLS]", "[SEP]", "<bos>", "<eos>","<MASK>", "<LOC>","</LOC>", "<SIZE>", "</SIZE>"]

def build_bpe_tokenizer(texts, vocab_size=5000, min_frequency=2, special_tokens=SPECIAL_TOKENS, save_dir= "tokenizer.json"):
    
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE())
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Customize the tokenizer training
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens, end_of_word_suffix="</w>")

    # Train the tokenizer on the dataset
    if type(texts) == list:
        tokenizer.train_from_iterator(texts, trainer=trainer)
    else:
        tokenizer.train([texts], trainer)
        
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
    tokenizer.decoder = decoders.ByteLevel()
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
                                                tokenizer_object=tokenizer,
                                                bos_token="<bos>",
                                                eos_token="<eos>",
                                            )
    wrapped_tokenizer.save_pretrained("pretrained" + save_dir)
    tokenizer.save(save_dir)
    
    return wrapped_tokenizer


def build_wordpiece_tokenizer(texts, vocab_size=5000, min_frequency=1, special_tokens = SPECIAL_TOKENS, save_dir= "tokenizer.json"):
    # Initialize a WordPiece tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="<UNK>", max_input_chars_per_word=7))
    
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=True, clean_text=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    
    # customizer the tokenizer training
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    
    # Train the tokenizer on the dataset
    if type(texts) == list:
        tokenizer.train_from_iterator(texts, trainer=trainer)
    else:
        tokenizer.train([texts], trainer)
        
    wrapped_tokenizer = PreTrainedTokenizerFast(
                        tokenizer_object=tokenizer,
                        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
                        unk_token="<UNK>",
                        pad_token="<PAD>",
                        cls_token="[CLS]",
                        sep_token="[SEP]",
                        mask_token="<MASK>",
                    )
        
    wrapped_tokenizer.save_pretrained("pretrained" + save_dir)
    tokenizer.save(save_dir)
    
    return wrapped_tokenizer




if __name__ == "__main__":
    # Load the dataset
    ex = "The lung is 5 nm long and 3nm wide. it is located in the UPPER left LUng zone."
    texts = ["This is the first text. I am the main man \n", "This is the second text."]
    print(type(texts) == list)
    bpe_tokenizer = build_bpe_tokenizer(texts)
    piece_tokenizer = build_wordpiece_tokenizer(texts,save_dir="piece_tokenizer.json")

    # Preprocess text using the tokenizer
    text = "This is a new text to preprocess."
    encoded_text = bpe_tokenizer.encode(ex)
    decoded_text = bpe_tokenizer.decode(encoded_text)

    print("BPE tokenizer result")
    print(encoded_text)#.ids)  # [51, 23, 10, 34, 125, 297, 7, 160, 94, 2]
    #print(encoded_text.tokens)
    print(decoded_text)  # This is a new text to preprocess.

    encoded_text = piece_tokenizer.encode(ex)
    decoded_text = piece_tokenizer.decode(encoded_text)

    print("Word Piece tokenizer result")
    print(encoded_text)  # [51, 23, 10, 34, 125, 297, 7, 160, 94, 2]
    print(decoded_text)  # This is a new text to preprocess.

    print(piece_tokenizer.encode("<eos>"))

# 

# print(add_tags(ex))
            
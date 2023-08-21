from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers import pre_tokenizers, processors, normalizers, decoders
from transformers import PreTrainedTokenizerFast

import json
import os
import re
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

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "[CLS]", "[SEP]", "<s>","<MASK>", "<size>", "<ind>", "<prompt>",
                  "T1", "T2","T3", "T4","T5","T6","T7","T8","T9","T10","T11","T12", "1","2","3","4","5","6","7","8","9","0",
                  "<loc0>", "<loc1>", "<loc2>", "<loc00>","<loc01>","<loc02>","<loc10>","<loc11>","<loc12>","<loc20>","<loc21>",
                  "<loc22>","<loc02><loc22>","<loc0><loc2>","<rib1>","<rib2>","<rib3>","<rib4>","<rib5>","<rib6>","<rib7>","<rib8>",
                  "<rib9>","<rib10>","<rib11>","<rib12>"]

#loc = axis, x, y
DIRECTIONS = {"left":"2", "right": "0", "up":"0", "mid":"1","down":"2", "mid-":"1","mid - ": "1", 
              "upper":"0", "lower":"2", "apical": "0","basilar": "2", "middle":"1", "bilateral": "<loc0><loc2>", "bibasal": "<loc02><loc22>","basal":"2",
              "anterior":"2", "bilaterally": "<loc0><loc2>", "bibasilar":"<loc02><loc22>", "posterior":"0","retrosternal":"2"}
NUMBERS = {"1st":"first", "2nd": "second", "3rd": "third", "4th": "fourth","5th": "fifth", "6th": "sixth","7th":"seventh",
           "8th": "eighth", "9th":"Ninth","10th":"tenth", "11th": "eleventh","12th": "twelfth"}


def build_bpe_tokenizer(texts, vocab_size=5000, min_frequency=2, special_tokens=SPECIAL_TOKENS, save_dir= "bpe_tokenizer.json"):
    
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
    #wrapped_tokenizer.save_pretrained(os.path.join("./pretrained", save_dir))
    tokenizer.save(save_dir)
    
    return wrapped_tokenizer


def build_wordpiece_tokenizer(texts, vocab_size=5000, min_frequency=1, special_tokens = SPECIAL_TOKENS, save_dir= "wordpiece_tokenizer.json"):
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
        
    #wrapped_tokenizer.save_pretrained(os.path.join("./pretrained", save_dir))
    tokenizer.save(save_dir)
    
    return wrapped_tokenizer



def contains_digit_and_alphabet(string):
    has_digit = False
    has_alpha = False
    
    for char in string:
        if char.isdigit():
            has_digit = True
        elif char.isalpha():
            has_alpha = True
        
        if has_digit and has_alpha:
            return True
    
    return False


def add_tags(text):
    #caption = caption.lower()
    text = re.sub("\s+"," ",text)
    #print(text)
    caption = text.replace('.',' . ')
    caption = caption.replace("\n"," \n ")
    caption = caption.replace(',', ' , ')
    caption = caption.replace(";"," ; ")
    caption_split = caption.split(" ")
    caption_length = len(caption_split)
    #print(caption_split)
    directions = []
    sizes = []
    #direction_start = False
    for index, word in enumerate(caption_split):
        if word.isdigit():
            sizes.append({word : "<size> " + word.strip(" ") })
                        
        elif contains_digit_and_alphabet(word):
            # number = ""
            # for char_ind, char in enumerate(word):
            #     if char_ind == 0 and not char.isdigit():
            #         break
            #     elif char.isdigit():
            #         number += char
            #     else:
            #         break
            # unit = word.strip(number).strip(" ")
            if word[0].isdigit() and not word.lower().endswith("th") or not word.lower().endswith("nd"):
                sizes.append({word :"<size> "  + word.strip(" ")})
            
        
        elif word.strip(" ").lower() in DIRECTIONS.keys():
            #direction_start = True
            
            if word.strip(" ").lower() == "bilateral" or word.strip(" ").lower() == "bilaterally":
                directions.append({word: DIRECTIONS[word.strip(" ").lower()] + " " + word})
                continue
            
            if word.strip(" ").lower() == "bibasilar":
                directions.append({word : DIRECTIONS[word.strip(" ").lower()] + " " + word})
                continue
            
            if index + 1 < caption_length:            
                if caption_split[index + 1].strip(" ").lower() not in DIRECTIONS.keys():
                    directions.append({word:  "<loc" + DIRECTIONS[word.strip(" ").lower()] + "> " + word})
                    #direction_start = False
                else:
                    dir = "<loc" + DIRECTIONS[word.strip(" ").lower()] + DIRECTIONS[caption_split[index + 1].strip(" ").lower()] + "> "
                    words = word.strip(" ") + " " + caption_split[index + 1].strip(" ")
                    # caption_split[index] = "<loc> " + word + " " + caption_split[index + 1] + " </loc>"
                    
                                    
                    # if  caption_split[index + 2] in DIRECTIONS.keys():
                    #     dir += dir + DIRECTIONS[caption_split[index + 2].lower()] +">"
                    #     caption_split.pop(index  + 2)

                    #     index +=4
                    # else:
                    caption_split.pop(index + 1)
                    #index += 3
                    #print(dir , words)
                    directions.append({words: dir + words})
            
        else:
            continue
    
    for each in sizes:    
        for item, value in each.items():
            text = text.replace(item, value,1)
    
    for each in directions:  
        for item, value in each.items():
            #print(item, value)

            #if item == "anterior middle" and item in text:
                #print("yes")
            text = text.replace(item, value,1)
        
    return text


if __name__ == "__main__":
    # Load the dataset
    ex = "The lung is 5 nm long and 3nm wide. t9 it is located in the middle of the UPPER left LUng zone. it can also be found in the  anterior middle \
        upper lung zone that you can tell me T12, 4th, anterior posterior"
    texts = ["This is the first text. I am the main man \n", "This is the second text."]
    # print(type(texts) == list)
    # bpe_tokenizer = build_bpe_tokenizer(texts)
    # piece_tokenizer = build_wordpiece_tokenizer(texts,save_dir="piece_tokenizer.json")

    # Preprocess text using the tokenizer
    text = "This is a new text to preprocess."
    # encoded_text = bpe_tokenizer.encode(ex)
    # decoded_text = bpe_tokenizer.decode(encoded_text)
    #print(add_tags(ex))
    # print("BPE tokenizer result")
    # print(encoded_text)#.ids)  # [51, 23, 10, 34, 125, 297, 7, 160, 94, 2]
    #print(encoded_text.tokens)
    #print(decoded_text)  # This is a new text to preprocess.

    # encoded_text = piece_tokenizer.encode(ex)
    # decoded_text = piece_tokenizer.decode(encoded_text)

    # print("Word Piece tokenizer result")
    # print(encoded_text)  # [51, 23, 10, 34, 125, 297, 7, 160, 94, 2]
    # print(decoded_text)  # This is a new text to preprocess.

    # print(piece_tokenizer.encode("<eos>"))
    
# Min tokens in captions: 0..
# Max tokens in captions: 60..
# Min tokens in indications: 2..
# Max tokens in indications: 51..
# Minium sentences: 1..
# Maximum sentences: 20..
# print(add_tags(ex))
            
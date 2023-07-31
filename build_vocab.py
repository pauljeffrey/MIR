import pickle
from collections import Counter
import json

DIRECTIONS = ["left", "right", "up", "mid","down", "upper", "lower","medial", "lateral", "middle", "contralateral", "bilateral"]
#RELATION

class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]
        # return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self, desc_tags=True, for_patient_info=False):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<unk>')
        
        if not for_patient_info:
            self.add_word('<end>')
            self.add_word('<start>')
        
        
        if desc_tags:
            self.add_word('<size>')
            self.add_word('</size>')
            self.add_word('<loc>')
            self.add_word('</loc>')
            
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold, desc_tags = True, save_dir= "vocab_tokenizer.pkl"):
    caption_reader = JsonReader(json_file)
    counter = Counter()

    for items in caption_reader:
        text = add_tags(items) #items.replace('.', ' . ').replace(',', ' , ')
        counter.update(text.lower().split(' '))
    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']
    vocab = Vocabulary(desc_tags)

    for word in words:
        #print(word)
        vocab.add_word(word)
        
    return vocab


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


def add_tags(caption):
    caption = caption.lower()
    caption = caption.replace('.',' . ')
    caption = caption.replace(',', ' , ')
    caption_split = caption.split(" ")
    #print(caption_split)
    for index in range(len(caption_split)):
        if index < len(caption_split):
            word = caption_split[index]
        else:
            return " ".join(caption_split)
        
        if word.isdigit():
            caption_split[index] = "<size> " + word + " </size>"
            
        elif contains_digit_and_alphabet(word):
            number = ""
            for char in word:
                if char.isdigit():
                    number += char
                else:
                    break
            unit = word.strip(number)
            caption_split[index] = "<size> "  + number + " </size> " +  unit
        
        elif word in DIRECTIONS:
            if caption_split[index + 1] not in DIRECTIONS:
                caption_split[index] = "<loc> " + word + " </loc>"
            else:
                caption_split[index] = "<loc> " + word + " " + caption_split[index + 1] + " </loc>"
                caption_split.pop(index + 1)
                index += 3
        
        else:
            continue
        
    return " ".join(caption_split)

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

def main(json_file, threshold, vocab_path):
    vocab = build_vocab(json_file=json_file,
                        threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))


if __name__ == '__main__':
    main(json_file='debugging_captions.json',
         threshold=0,
         vocab_path='debug_vocab.pkl')
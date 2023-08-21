import pickle
from collections import Counter
import json


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
        self.add_word("<PAD>" )
        self.add_word("<UNK>")
        
        if not for_patient_info:
            self.add_word('</s>')
            self.add_word('<start>')
       
        if desc_tags:
            self.add_word('<size>')
            #self.add_word('</size>')
            self.add_word('<loc>')
            #self.add_word('</loc>')
            
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
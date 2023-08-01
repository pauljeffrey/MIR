import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from build_vocab import Vocabulary, JsonReader
import numpy as np
from torchvision import transforms
import pickle
from text_preprocessing import *


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 history_json,
                 file_list,
                 vocabulary,
                 vocabulary2,
                 s_max=10,
                 n_max=50,
                 transforms=None):
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)
        self.history = JsonReader(history_json)
        self.file_names, self.labels = self.__load_label_list(file_list)
        self.vocab = vocabulary
        self.vocab2 = vocabulary2
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max

    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = '{}.png'.format(image_name)
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        try:
            text = self.caption[image_name]
            history = self.history[image_name]
        except Exception as err:
            text = 'normal. '

        target = list()
        max_word_num = 0
        for i, sentence in enumerate(text.split('. ')):
            if i >= self.s_max:
                break
            sentence = sentence.split()
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token) for token in sentence])
            tokens.append(self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        
        history_max_word_num = 0
        tokens = [self.vocab2(token) for token in history.split()]
        if history_max_word_num < len(tokens):
            history_max_word_num = len(tokens)
        
        
        return image, tokens, label, target, sentence_num, max_word_num, history_max_word_num

    def __len__(self):
        return len(self.file_names)


def collate_fn(data):
    images, tokens, label, captions, sentence_num, max_word_num, history_max_word_num = zip(*data)
    images = torch.stack(images, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)
    
    history_max_word_num = max(history_max_word_num)

    patient_history = np.zeros((len(tokens), history_max_word_num))
    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    for i, token in enumerate(tokens):
        patient_history[i,len(token)] = token[:]
        
    return images, patient_history, torch.Tensor(label), targets, prob


def collate_fn2(data):
    images, tokens, _, captions, sentence_num, max_word_num, history_max_word_num = zip(*data)
    images = torch.stack(images, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)

    history_max_word_num = max(history_max_word_num)

    patient_history = np.zeros((len(tokens), history_max_word_num))
    
    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    for i, token in enumerate(tokens):
        patient_history[i,len(token)] = token[:]
        
    return images, patient_history, targets, prob


def get_loader(image_dir,
               caption_json,
               history_json,
               file_list,
               vocabulary,
               vocabulary2,
               transform,
               batch_size,
               s_max=10,
               n_max=50,
               shuffle=True,
               collate_fn=collate_fn
               ):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               caption_json=caption_json,
                               history_json = history_json,
                               file_list=file_list,
                               vocabulary=vocabulary,
                               vocabulary2 = vocabulary2,
                               s_max=s_max,
                               n_max=n_max,
                               transforms=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


class ChestXrayDataSet2(Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 file_list,
                 tokenizer_name,
                 s_max=10,
                 n_max=50,
                 transforms=None,
                 use_tokenizer_fast =True):
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)
        self.file_names, self.labels = self.__load_label_list(file_list)
        if use_tokenizer_fast:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = Tokenizer.from_file(tokenizer_name)
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max

    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = '{}.png'.format(image_name)
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        try:
            text = self.caption[image_name]
        except Exception as err:
            text = 'normal. '

        target = list()
        max_word_num = 0
        for i, sentence in enumerate(text.split('. ')):
            if i >= self.s_max:
                break
            sentence = self.tokenizer.encode(sentence).ids
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue
            tokens = list()
            tokens.extend(self.tokenizer.encode('<bos>'))
            tokens.extend(sentence)
            tokens.extend(self.tokenizer.encode('<eos>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        return image, image_name, label, target, sentence_num, max_word_num

    def __len__(self):
        return len(self.file_names)



def get_loader2(image_dir,
               caption_json,
               file_list,
               tokenizer_name,
               transform,
               batch_size,
               s_max=10,
               n_max=50,
               shuffle=True,
               use_tokenizer_fast=True,
               collate_fn=collate_fn2
               ):
    
    dataset = ChestXrayDataSet2(image_dir=image_dir,
                               caption_json=caption_json,
                               file_list=file_list,
                               tokenizer_name=tokenizer_name,
                               s_max=s_max,
                               n_max=n_max,
                               transforms=transform,
                               use_tokenizer_fast=use_tokenizer_fast)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


class EncoderDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 file_list,
                 transforms=None):
        self.image_dir = image_dir
        self.file_names, self.labels = self.__load_label_list(file_list)
        self.transform = transforms
        
    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = '{}.png'.format(image_name)
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name, label

    def __len__(self):
        return len(self.file_names)




def get_enc_loader(image_dir,
               file_list,
               transform,
               batch_size,
               shuffle=True):
    
    dataset = EncoderDataSet(image_dir=image_dir,
                               file_list=file_list,
                               transforms=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=None)
    return data_loader


if __name__ == '__main__':
    vocab_path = '../data/vocab.pkl'
    image_dir = '../data/images'
    caption_json = '../data/debugging_captions.json'
    file_list = '../data/debugging.txt'
    batch_size = 6
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             caption_json=caption_json,
                             file_list=file_list,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=False)

    for i, (image, image_id, label, target, prob) in enumerate(data_loader):
        print(image.shape)
        print(image_id)
        print(label)
        print(target)
        print(prob)
        break
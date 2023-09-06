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
import random
import numpy as np
from clean_caption import *
from omegaconf import OmegaConf
from torchvision import transforms

import cProfile
import io
import pstats
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


def collate_fn2(data): #, history_word_num=60
    indication, captions, sentence_num, word_num = zip(*data)  #labels,  images,
    #images = torch.stack(images, 0)
    #print("In the collate_fn...")
    #print(labels.shape)
    #labels = torch.stack(labels, 0).type(torch.LongTensor)
    max_prompt_length = max([len(each) for each in indication])
    max_word_num = max(word_num)
    max_sentence_num = max(sentence_num)
    #max_word_num = max(max_word_num)

    #history_max_word_num = max(history_max_word_num)

    indication_prompts = np.zeros((len(indication), max_prompt_length))
    
    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    probs = np.ones((len(captions), max_sentence_num + 1))  * -1

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence
            # if len(sentence) > 0:
            probs[i,j] = 0
            
        probs[i,len(caption)] = 1
            
    for i, tokens in enumerate(indication):
        indication_prompts[i,:len(tokens)] = tokens
        
    indication_prompts = torch.tensor(indication_prompts).type(torch.LongTensor)
    probs = torch.tensor(probs).type(torch.LongTensor)
    targets = torch.tensor(targets).type(torch.LongTensor)
    
    #print(type(prompt), type(label), type(target), type(prob))
    del indication 
    del captions
    del sentence_num
    del word_num
    del max_word_num
    del max_prompt_length
    del max_sentence_num
    
    return  indication_prompts , probs, targets #images,  labels,


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
               shuffle=False,
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
                 tokenizer_name,
                 s_max=15,
                 n_max=40,
                 encoder_n_max=60,
                 transforms=None,
                 use_tokenizer_fast =True):
        
        self.image_dir = image_dir
        if caption_json.endswith("val.json"):
            with open(caption_json, 'r') as f:
                self.data = random.sample(json.load(f), 1000)
        else:
            # with open(caption_json, 'r') as f:
            #     self.data  = json.load(f)
            self.data = pd.read_json(caption_json).values
            #print(self.data.columns)
        #self.file_names, self.labels = self.__load_label_list(file_list)
        if use_tokenizer_fast:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        else:
            #print(tokenizer_name)
            self.tokenizer = Tokenizer.from_file(tokenizer_name)
            
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max
        self.encoder_n_max = encoder_n_max

    def __getitem__(self, index):
        #['image', 'type', 'caption', 'problems', 'indication', 'labels']
        sample = self.data[index]
        #image_name = sample[0] #sample["image"]
        # if index > 1200:  
        #     print("In the dataset function...")
        # #image = 
        #print("Image shape: ", image.size)
        #label = torch.tensor([int(each) for each in sample["labels"]])
        
        if sample[1] == "original":
            
            indication = sample[4] #sample["indication"]
            if "<prompt>" in indication:
                indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + indication.split("<ind>")[-1]
                indication = indication.split("<prompt>")[0] + "<prompt>" + add_noise(indication.split("<prompt>")[1]) + "<prompt>"
                
            else:
                indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>"
        else:
            #label = torch.ones((len(sample["labels"]))) * -1
            indication = sample[4]  #sample["indication"]
            indication = indication.split("<prompt>")[0] + "<prompt>" + add_noise(indication.split("<prompt>")[1]) + "<prompt>"
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + indication.split("<ind>")[-1]
            if "<prompt>" in indication:
                indication = rm_indication(indication)
                
            
        # if self.transform is not None:
        #     if index % 2 == 0:
        #         image = self.transform(Image.open(os.path.join(self.image_dir, image_name)).convert('RGB'))
        #     else:
        #         image = self.transform(Image.open(os.path.join(self.image_dir, image_name)).convert('RGB'))
            
        caption = sample[2] #sample["caption"]
        
        
        target = list()
        indication_prompt = list()
        word_num = 0
        #max_word_num = 0
        for i, sentence in enumerate(caption.split('.')):
            if i >= self.s_max:
                break
            
            if len(sentence) == 0 or (len(sentence) == 1 and sentence in [".",",",";",":","@","/","-","_","%","*"]):
                continue
            sentence = self.tokenizer.encode(sentence).ids
            if len(sentence) > self.n_max:
                sentence = sentence[:self.n_max]
                
            tokens = list()
            tokens.extend(self.tokenizer.encode('<s>').ids)
            tokens.extend(sentence)
            tokens.extend(self.tokenizer.encode('<s>').ids)
            # if max_word_num < len(tokens):
            #     max_word_num = len(tokens)
            word_num = max(word_num, len(tokens))
            target.append(tokens)
            
        sentence_num = len(target)
        
        indication_prompt.extend(self.tokenizer.encode(indication).ids)
        
        if len(indication_prompt) > self.encoder_n_max:
            indication_prompt = indication_prompt[:self.encoder_n_max -2] + self.tokenizer.encode('<prompt>').ids
        
        return  indication_prompt, target, sentence_num, word_num  #image_name,label,  image,

    def __len__(self):
        return len(self.data)



def get_loader2(image_dir,
               caption_json,
               tokenizer_name,
               transform,
               batch_size=8,
               s_max=15,
               n_max=40,
               encoder_n_max=60,
               shuffle=True,
               use_tokenizer_fast=False,
               collate_fn=collate_fn2
               ):
    
    dataset = ChestXrayDataSet2(image_dir=image_dir,
                               caption_json=caption_json,
                               tokenizer_name=tokenizer_name,
                               s_max=s_max,
                               n_max=n_max,
                               encoder_n_max=encoder_n_max,
                               transforms=transform,
                               use_tokenizer_fast=use_tokenizer_fast)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn,
                                              num_workers = 0,
                                              pin_memory=True)
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
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    transform = transforms.Compose(
    [
        transforms.Resize((224,224), antialias=True),        
        transforms.ToTensor(),
    ]
    )
    cfg = OmegaConf.load("/kaggle/working/MIR/conf/config.yaml") #
    train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.train_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn2)
    
    
    def check(train_loader):
        for step, ( indication_prompt, true_stop_probs, reports) in enumerate(train_loader): #encoded_images,
            if step <= 20000:
                print( indication_prompt.shape, true_stop_probs.shape, reports.shape) #encoded_images.shape,
            else:
                break
        return 

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("check"):
            check(train_loader)

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

        
    # vocab_path = '../data/vocab.pkl'
    # image_dir = '../data/images'
    # caption_json = '../data/debugging_captions.json'
    # file_list = '../data/debugging.txt'
    # batch_size = 6
    # resize = 256
    # crop_size = 224

    # transform = transforms.Compose([
    #     transforms.Resize(resize),
    #     transforms.RandomCrop(crop_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))])

    # with open(vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)

    # data_loader = get_loader2(image_dir="jeefff",
                              
    #                          caption_json="./data/full_data/train.json",
    #                          tokenizer_name= "./tokenizers/wordpiece_tokenizer8000.json",
    #                          transform=None,
    #                          batch_size=8,
    #                          use_tokenizer_fast=False,
    #                          shuffle=True)

    # for i, (prompt, label,  prob, target) in enumerate(data_loader):
        
    #     print(prompt.shape)
    #     print(label.shape)
    #     print(target.shape)
    #     print(prob.shape)
    #     print(prob)
    #     print(type(prompt), type(label), type(target), type(prob))
    #     break
    
    # with open("./data/full_data/val.json", "r") as f:
    #     print(len(json.load(f)))
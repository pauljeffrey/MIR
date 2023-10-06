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
from typing import Union
#from multiprocessing import Manager, Array
#import cstl
import psutil

from torch.utils.data import WeightedRandomSampler

from multiprocessing import shared_memory #import ShareableList
import cProfile
import io
import pstats
import pickle
from typing import List,Any

class NumpySerializedList():
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)
    

class ChestXrayDataSet(Dataset):
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
        if caption_json.endswith("validation.json") or caption_json.endswith("val.json"):
            with open(caption_json, "r") as f:
                data = json.load(f) 
                for _ in range(20):
                    data = random.sample(data, len(data))
                
                data = random.sample(data, 2048)
            
        else:          
            with open(caption_json, "r") as f:
                data = json.load(f)
                #data.reverse()
            
            #data = pd.read_json(caption_json)#, 'type', "caption","indication"
          
        self.len = len(data)
        print("Chestxray Dataset..")

        self._data = NumpySerializedList(data)
        
        if use_tokenizer_fast:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        else:
            #print(tokenizer_name)
            self.tokenizer = Tokenizer.from_file(tokenizer_name)
            
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max
        self.encoder_n_max = encoder_n_max
        
    #@profile
    def __getitem__(self, index):      
        sample = self._data[index]
        # print(f"There are {len(psutil.process_iter())} processes running in the dataset __getitem__()")
        # for process in psutil.process_iter():
        #     print(f"Process Name: {process.name}, Process ID: {process.pid}.")
            
        image_name = sample["image"]
        indication = sample["indication"]
        caption = sample["caption"]
        #print(f"Index number {index}- Image name: {image_name}")
        # if sample_type == "original":
            
            #indication = sample[4] #sample["indication"]
        if "<prompt>" in indication:
            indication = indication.split("<prompt>")[0] + "<prompt>" + add_noise(indication.split("<prompt>")[1]) + "<prompt>"
            indication = rm_indication(indication)
            if "<ind>" in indication:
                indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + indication.split("<ind>")[-1]
                
            
        else:
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + add_prompt()
       
        if self.transform is not None:
            if index % 2 == 0:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            else:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            
        
        caption = [self.tokenizer.encode(sent).ids[:self.n_max] for sent in caption.split('.')[:self.s_max] if not 
                   (len(sent) == 0 or (len(sent) == 1 and sent in [".",",",";",":","@","/","-","_","%","*"]))]
        
        max_word_num = 0
        
        for each in caption:
            #print(type(each))
            each.insert(0, self.tokenizer.encode('<s>').ids[0])
            each.append(self.tokenizer.encode('<s>').ids[0])
            #each.extend([0] * (self.n_max - len(each)))
            max_word_num = max(max_word_num, len(each))
            
        
        indication = self.tokenizer.encode(indication).ids
        if len(indication) > self.encoder_n_max:
            #print("This is bigger: ", len(indication))
            indication = indication[:self.encoder_n_max - 1] + self.tokenizer.encode('<prompt>').ids 
        
        return  image , indication, caption , len(caption), max_word_num  #image_name,label,  image,

    def __len__(self):
        return self.len



class ChestXrayDataSet3(Dataset):
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
          
        df = pd.read_json(caption_json)
        self.dataset  = Dataset.from_pandas(df)

          
        self.len = len(df)
        print("Chestxray Dataset..")
        
        
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
        sample = self.dataset[index] 
        
        
        image_name = sample["image"]
        indication = sample["indication"]
        caption = sample["caption"]
        # if sample_type == "original":
            
            #indication = sample[4] #sample["indication"]
        if "<prompt>" in indication:
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + indication.split("<ind>")[-1]
            indication = indication.split("<prompt>")[0] + "<prompt>" + add_noise(indication.split("<prompt>")[1]) + "<prompt>"
            indication = rm_indication(indication)
            
        else:
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>"
       
        if self.transform is not None:
            if index % 2 == 0:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            else:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            
        
        caption = [self.tokenizer.encode(sent).ids[:self.n_max] for sent in caption.split('.')[:self.s_max] if not 
                   (len(sent) == 0 or (len(sent) == 1 and sent in [".",",",";",":","@","/","-","_","%","*"]))]
        
        max_word_num = 0
        
        for each in caption:
            #print(type(each))
            each.insert(0, self.tokenizer.encode('<s>').ids[0])
            each.append(self.tokenizer.encode('<s>').ids[0])
            #each.extend([0] * (self.n_max - len(each)))
            max_word_num = max(max_word_num, len(each))
            
        
        indication = self.tokenizer.encode(indication).ids
        if len(indication) > self.encoder_n_max:
            #print("This is bigger: ", len(indication))
            indication = indication[:self.encoder_n_max - 1] + self.tokenizer.encode('<prompt>').ids 
        
        return  image , indication, caption , len(caption), max_word_num  #image_name,label,  image,

    def __len__(self):
        return self.len



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

#@profile
def collate_fn2(data): #, history_word_num=60
    images, indication, captions, sentence_num, word_num = zip(*data)  #labels,  
    #print(len(images), len(indication), len(captions), len(sentence_num,len(word_num)))
    images = torch.stack(images, 0)  
    #New
    #max_ind = max([len(each) for each in indication])
    #max_sent = max([len(each) for each in captions])
    #indication = np.array([ each.extend([0] * (max_ind - len(each))) for each in indication] , dtype="float16")

    indication_prompts = np.zeros((len(indication), max([len(each) for each in indication])))
    
    targets = np.zeros((len(captions), max(sentence_num) + 1, max(word_num)))
    
    probs = np.ones((len(captions), max(sentence_num) + 1), )  * -1

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence
            # if len(sentence) > 0:
            probs[i,j] = 0
            
        probs[i,len(caption)] = 1
            
    for i, tokens in enumerate(indication):
        indication_prompts[i,:len(tokens)] = tokens
        
    indication_prompts = torch.tensor(indication_prompts).type(torch.LongTensor)#.to("cuda")
    probs = torch.tensor(probs).type(torch.LongTensor)#.to("cuda")
    targets = torch.tensor(targets).type(torch.LongTensor)#.to("cuda")
    #print(indication)
    del indication
    del captions
    del sentence_num
    del word_num
    
    return  images ,indication_prompts , probs, targets #images,  labels,


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

def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])
    

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

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
        if caption_json.endswith("validation.json") or caption_json.endswith("val.json"):
            data = pd.read_json(caption_json)
                
            data = data.iloc[:250]
            
        else:  
            data = pd.read_json(caption_json)#, 'type', "caption","indication"
            
        self.len = len(data)
            
        seqs = [string_to_sequence(s) for s in data["image"]]
        self.images_v, self.images_o = pack_sequences(seqs)
        
        seqs = [string_to_sequence(s) for s in data["caption"]]
        self.captions_v, self.captions_o = pack_sequences(seqs)
        
        seqs = [string_to_sequence(s) for s in data["indication"]]
        self.indications_v, self.indications_o = pack_sequences(seqs)
        
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
        seq = unpack_sequence(self.images_v, self.images_o, index)
        
        image_name = sequence_to_string(seq) 
        seq = unpack_sequence(self.captions_v, self.captions_o, index)
        caption = sequence_to_string(seq)
        
        seq = unpack_sequence(self.indications_v, self.indications_o, index)
        indication = sequence_to_string(seq)
        #indication, caption = caption.split("<<END>>")       
         
        del seq
        # if sample_type == "original":
            
            #indication = sample[4] #sample["indication"]
        if "<prompt>" in indication:
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>" + indication.split("<ind>")[-1]
            indication = indication.split("<prompt>")[0] + "<prompt>" + add_noise(indication.split("<prompt>")[1]) + "<prompt>"
            indication = rm_indication(indication)
            
        else:
            indication = "<ind>" + add_noise(indication.split("<ind>")[1]) + "<ind>"
       
        if self.transform is not None:
            if index % 2 == 0:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            else:
                image = self.transform(Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB'))
            
        
        caption = [self.tokenizer.encode(sent).ids[:self.n_max] for sent in caption.split('.')[:self.s_max] if not 
                   (len(sent) == 0 or (len(sent) == 1 and sent in [".",",",";",":","@","/","-","_","%","*"]))]
        
        max_word_num = 0
        
        # New  
        # probs = np.ones( self.s_max + 1)  * -1
        # probs[:len(caption)] = 1
        # probs[len(caption)] = 0
        
        for each in caption:
            #print(type(each))
            each.insert(0, self.tokenizer.encode('<s>').ids[0])
            each.append(self.tokenizer.encode('<s>').ids[0])
            #each.extend([0] * (self.n_max - len(each)))
            max_word_num = max(max_word_num, len(each))
            
        # New    
        # for _ in range(self.s_max - len(caption)):
        #     caption.append([0] * self.n_max)
        #     #max_word_num = max(max_word_num, len(each))
        
        indication = self.tokenizer.encode(indication).ids
        # #indication_prompt.extend(self.tokenizer.encode(indication).ids)
        #print("Indication before padding: ", indication)
        if len(indication) > self.encoder_n_max:
            #print("This is bigger: ", len(indication))
            indication = indication[:self.encoder_n_max - 1] + self.tokenizer.encode('<prompt>').ids 
            #print("max: ", len(indication))
        # elif len(indication) < self.encoder_n_max:
        #     indication.extend([0]* (self.encoder_n_max - len(indication)))
        #     #print("min: ", len(indication))
            
        # indication = torch.tensor(indication)
        # probs = torch.tensor(probs)
        # caption = torch.tensor(caption)
        
        return  image , indication, caption , len(caption), max_word_num  #image_name,label,  image,

    def __len__(self):
        return self.len



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
               collate_fn=collate_fn2,
               #sampler = None
               ):
    
    dataset = ChestXrayDataSet(image_dir=image_dir,
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
                                              #drop_last = False,
                                              collate_fn=collate_fn,
                                              num_workers = 8,
                                              #sampler=sampler,
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
    import gc

    transform = transforms.Compose(
    [
        # transforms.RandomRotation((0,5)),
        # #transforms.v2.RandomResize((200, 250)), v2.RandomResize
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
        # transforms.ColorJitter(brightness= (0.5, 1.5) , contrast=(0, 1.0)),
        # transforms.Pad(20),
        transforms.Resize((224,224), antialias=True), 
        transforms.ToTensor(),
    ]
    )
    
    # with open("/kaggle/working/weights.json", "r") as f:
    #     weights = json.load(f)
  
    # sampler = WeightedRandomSampler(
    #     weights=weights,
    #     num_samples=len(weights),
    #     replacement=False
    # )
    
    cfg = OmegaConf.load("/content/sample_data/MIR/conf/config.yaml") #
    train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.train_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, 
            collate_fn=collate_fn2)#, sampler=sampler)
    
    #print(cfg.dataset.train.caption_json)
    #def check(train_loader):
    for step, (images ,indication_prompts , probs, targets) in enumerate(train_loader): #encoded_images, indication_prompt, true_stop_probs, reports
        if step % 500 == 0: 
            print(step, images.shape, indication_prompts.shape, probs.shape, targets.shape) #encoded_images.shape, indication_prompt.shape, true_stop_probs.shape, reports.shape
            print("Instruction prompts: ", indication_prompts)
            print("Stop_prob: ", probs)
            print("Targets: ", targets)
         #,indication_prompt,reports
        gc.collect()


    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("check"):
    #         check(train_loader)

    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

        
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
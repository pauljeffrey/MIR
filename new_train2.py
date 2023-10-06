
from torch import nn
from utils import *
from loss import *
from dataset import *
#from build_vocab import load_vocab
from full_model import * 

from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup

from accelerate import Accelerator
from accelerate import Accelerator, DistributedType ,DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
import logging
import math
import os
import torch
#import hydra
import torch

from tqdm.auto import tqdm
from transformers import (
    get_scheduler,
)
import torch.autograd.profiler as profiler
from torch.utils.data import WeightedRandomSampler
from time import time

torch.manual_seed(42)

# def save_model(model , save_dir):
#     if not os.path.exists("save_dir"):
#         os.mkdir(save_dir)
        
#     path = os.path.abspath(save_dir)
    
#     # Save each module
#     if model.encoder is not None:
#         torch.save(model.encoder.state_dict(), os.path.join(path, 'encoder.pt'))
    
#     if model.history_encoder is not None:
#         torch.save(model.history_encoder.state_dict(), os.path.join(path, "prompt_encoder.pt"))
        
#     if model.semantic_features_extractor is not None:
#         torch.save(model.semantic_features_extractor.state_dict(), os.path.join(path, "semantic_features_extractor.pt"))
        
#     torch.save(model.attention.state_dict(), os.path.join(path, "attention.pt"))
#     torch.save(model.sent_lstm.state_dict(), os.path.join(path, "sent_lstm.pt"))
#     torch.save(model.decoder.state_dict(), os.path.join(path, "decoder.pt"))

#     return

def save_model(model, optimizer= None, lr_scheduler=None, epoch=None, step = None, loss=None, path="./model.tar"):
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    if optimizer is not None: 
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "prompt_encoder": model.history_encoder.state_dict(),
                "attention": model.attention.state_dict(),
                "sent_lstm": model.sent_lstm.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler" : lr_scheduler.state_dict(),
                "loss": loss ,
                "epoch": epoch,
                "last_step": step
            }, os.path.join(path, "checkpoint.tar")
        )
        
    else:
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "prompt_encoder": model.history_encoder.state_dict(),
                "attention": model.attention.state_dict(),
                "sent_lstm": model.sent_lstm.state_dict(),
                "decoder": model.decoder.state_dict(),
            }, os.path.join(path, "checkpoint.tar")
        )
        
    return

def load(model, cfg, from_checkpoint=False):
    checkpoint = torch.load(os.path.join(cfg.output_dir, cfg.load_dir, "checkpoint.tar"))
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.history_encoder.load_state_dict(checkpoint["prompt_encoder"])
    model.attention.load_state_dict(checkpoint["attention"])
    model.sent_lstm.load_state_dict(checkpoint["sent_lstm"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    
    if from_checkpoint:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if "lr_scheduler" in checkpoint.keys():
            lr_scheduler = get_scheduler(
                            name=cfg.training.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.training.lr_warmup_steps,
                            num_training_steps=cfg.training.max_train_steps,
                        )
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        else:
            lr_scheduler = get_scheduler(
                            name=cfg.training.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.training.lr_warmup_steps,
                            num_training_steps=cfg.training.max_train_steps,
                        )
            
        epoch = checkpoint['epoch']
        
        if "last_step" in checkpoint.keys():
            last_step = checkpoint['last_step']
        else:
            last_step = None
            
        loss = checkpoint['loss']
        return model, optimizer, lr_scheduler, epoch, last_step, loss
    else:        
        return model

#@hydra.main(version_base=None, config_path="conf", config_name="config")
def load_model(cfg: DictConfig, device= "cuda"):
    model_params = cfg.architecture
    #model_params.device = device
    model = MedicalReportGenerator(**model_params).to(device)
    if cfg.model.from_trained:
        path = os.path.join(os.path.abspath(cfg.output_dir), cfg.load_dir) # Assumes that all state_dicts are stored in the same directory.
        print(f"\nLoading all sub model weights from {path}...")
        model = load(model, cfg)
        return model
    elif cfg.model.from_checkpoint:
        path = os.path.join(os.path.abspath(cfg.output_dir ), cfg.load_dir) # Assumes that all state_dicts are stored in the same directory.
        print(f"\nLoading checkpoint from {path}...")
        model, optimizer, lr_scheduler, epoch, last_step, loss = load(model, cfg, True)
        return model, optimizer, lr_scheduler, epoch, last_step, loss
    else:
        return model
        
    

def create_optimizer(cfg, model):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": model.encoder.parameters(),
            "weight_decay": cfg.training.weight_decay,
            "lr": 8e-6
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("encoder")
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, accelerator, eval_loader, custom_loss): #, bce_loss
    model.eval()
    device = accelerator.device
    #loss = 0
    #print("\n In the evaluation function")
    with torch.no_grad():
        eval_losses = []
        eval_stop_losses = []
        similarity_losses = []
        #eval_bce_losses = []
        print("In evaluation function ....")
        for _ , (encoded_images, indication_prompt, true_stop_probs, reports) in enumerate(eval_loader):   #labels,      
            #n_sentences  = reports.shape[1]
            n_sentences  = reports.shape[1]
            # print('Length of sentences: ', n_sentences)
            # print("Inputs shape: ", encoded_images.shape, indication_prompt.shape, labels.shape, true_stop_probs.shape, reports.shape)
            # print("True probs: ",true_stop_probs)
            
            encoder_pad_mask = create_padding_mask(indication_prompt).to(device)
            #print("Mem shape: ", indication_prompt.shape, "mask shape: ", encoder_pad_mask.shape)
            #encoder_causal_mask = src_mask(indication_prompt.shape[1])
            
            encoded_images  = model.encoder(encoded_images)#.type(torch.cuda.HalfTensor)) # , tags
            
            if torch.any(torch.isinf(encoded_images)) or torch.any(torch.isnan(encoded_images)):
                print("Encoded Images is nan")
            
            # if torch.any(torch.isinf(tags)) or torch.any(torch.isnan(tags)):
            #     print("Tags are nan")
            #print("Encoded Images: ", encoded_images.shape)
            bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
            
            # if model.co_attention:
            #     semantic_features = model.semantic_features_extractor(tags)
                
            #if model.history_encoder is not None:
            indication_prompt = model.decoder.embed_layer(indication_prompt)
            # if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
            #     print("Encoding by decoder embedding layer is nan")
                
            indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask.type(indication_prompt.dtype))
            if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
                print("Encoding by History encoder is nan")  

            # # compute mask, confirm the first part.
            # mem_mask = torch.cat([torch.zeros((encoded_images.shape[0], encoded_images.shape[1])), encoder_pad_mask], dim=-1)
            
            #Initialize states
            if len(encoded_images.shape) > 3:
                encoded_images = encoded_images.reshape(bs, n_channels, -1)
                
            #print("lstm hidden state and cell state: ", model.sent_lstm.num_layers)
            
            prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt) #prev_hidden, 
            
            #lstm_init = True
            #print(hn.shape, cn.shape)
            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
                    print("Initial Prev hidden, hn, cn is nan")
                    
            for i in range(n_sentences):
                ##print(f"This loop is for the number {i} sentence.")
                
                for name , each in model.named_parameters():
                    if torch.any(torch.isnan(each)):
                        print(name, " layer has nan values in it..")
                    
                # if model.co_attention:
                #     if model.history_encoder is not None:
                #         #print("Encoded Images before attentino: ", encoded_images.shape)
                #         context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, indication_prompt)
                #     else:
                #         context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                # else:
                
                # Attend to encoded_images
                #if model.history_encoder is not None:
                context_vector, att_wts = model.attention(prev_hidden, encoded_images)    #prev_hidden, encoded_images, indication_prompt                      
                #else:
                #context_vector, att_wts = model.attention(prev_hidden, encoded_images)

                # if torch.any(torch.isinf(context_vector)) or torch.any(torch.isnan(context_vector)):
                #     print("Context Vector is nan")
                    
                # Generate Topic Vector
                #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
                prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
                #lstm_init= False
                if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
                    print("New prev hidden is nan")
                    
                if torch.any(torch.isinf(hn)) or torch.any(torch.isnan(hn)):
                    print("hn , cn are nan")
                    
                if torch.any(torch.isinf(pred_stop_probs)) or torch.any(torch.isnan(pred_stop_probs)):
                    print("Pred_stop_probs is nan")
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                #print("Target indices: ", tgt)
                #print('Target mask: ', tgt.shape)
                #padding_mask = create_padding_mask(tgt).to(device).type(indication_prompt.dtype)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1]).to(device).type(indication_prompt.dtype)
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None
                if torch.any(torch.isinf(memory)) or torch.any(torch.isnan(memory)):
                    print("Memory is affected contains nan values..")
                    
                if torch.any(torch.isnan(encoder_pad_mask)) or torch.any(torch.isnan(tgt_mask)):#torch.any(torch.isnan(padding_mask)) or 
                    print("Padding mask contains nan values...")
                # Compute attention for visual_features, encoded_prompt
                #memory = model.prompt_attention(memory, indication_prompt, key_padding_mask=mem_mask, residual_connection=True)
                # print(memory.shape, indication_prompt.shape, tgt.shape, prev_hidden.shape)
                # print(encoder_pad_mask.shape)
                output, final_decoder_layer_output = model.decoder(tgt, prev_hidden, (indication_prompt, memory), tgt_key_padding_mask=None,
                                    memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
                                        tgt_is_causal=False)  # [batch_size, seq_len - 1, d_model] 
                
                # if torch.any(torch.isinf(output)) or torch.any(torch.isnan(output)):
                #     print("Output is affected...")
                    
                # print("output shape: ", output.shape, reports[:, i, 1:].shape)
                # print("stop prob shape: ", pred_stop_probs.shape, true_stop_probs[:, 0].shape)
                
                #loss += custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:], pred_stop_probs,  output, eval=True)  # Ignore <sos> token
                stop_loss, sparse_loss, similarity_loss = custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:],
                                                     pred_stop_probs,  output, prev_hidden, final_decoder_layer_output, eval=True)  # Ignore <sos> token
                
                # print("stop loss: ", stop_loss)
                # print("sparse_loss: ", sparse_loss)
                eval_stop_losses.append(accelerator.gather(stop_loss.repeat(cfg.training.eval_batch_size)))
                eval_losses.append(accelerator.gather(sparse_loss.repeat(cfg.training.eval_batch_size)))
                similarity_losses.append(accelerator.gather(similarity_loss.repeat(cfg.training.eval_batch_size)))
                
            # print("loss : ", stop_loss, sparse_loss)
            # print("Loss list: ",eval_stop_losses, eval_losses)

            #binary_loss = bce_loss(tags, labels)
            #eval_bce_losses.append(accelerator.gather(binary_loss).detach().cpu())
            

        try:
            eval_loss = torch.mean(torch.cat(eval_losses))
            eval_stop_loss = torch.mean(torch.cat(eval_stop_losses))
            dist_loss = torch.mean(torch.cat(similarity_losses))
            #eval_bce_loss = torch.mean(torch.cat(eval_bce_losses))
            perplexity = math.exp(eval_loss.item())
            #print("Try : ", eval_loss, eval_stop_loss)    
            
        except OverflowError:
            perplexity = float("inf")
        
        #print(eval_loss, eval_stop_loss, perplexity)
                    
    return eval_loss , eval_stop_loss, perplexity, dist_loss #eval_bce_loss


#@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    #torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=cfg.training.gradient_accumulation_steps, offload_param_device= 'cpu', 
                                      offload_optimizer_device="cpu",gradient_clipping=1.0)
    accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin =deepspeed_plugin) #,  mixed_precision='fp16',
    
    accelerator.wait_for_everyone()
    device= accelerator.device
    #print(device)

    # logger.info(accelerator.state, main_process_only=False)
    # logger.info(OmegaConf.to_yaml(cfg))
    
    if not cfg.model.from_checkpoint:
        model = load_model(cfg, device= device)
        epoch = None
        last_step = None
        loss = None
        # Optimizer
        # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
        optimizer_cls = (
            torch.optim.AdamW #Adafactor #
            # if accelerator.state.deepspeed_plugin is None
            # or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            # else DummyOptim
        )
        optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)


    #     # if (
    #     #     accelerator.state.deepspeed_plugin is None
    #     #     or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    #     # ):
    else:
        model, optimizer, lr_scheduler, epoch, last_step, loss = load_model(cfg)
        # Remove this later
        last_step = None
        
    logger.info(f"Model with {count_parameters(model)} parameters loaded ...")
     
    # if lr_scheduler is None:   
    #     lr_scheduler = get_scheduler(
    #         name=cfg.training.lr_scheduler,
    #         optimizer=optimizer,
    #         num_warmup_steps=cfg.training.lr_warmup_steps,
    #         num_training_steps=cfg.training.max_train_steps,
    #     )
    # else:
    #     lr_scheduler = DummyScheduler(
    #         optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
    #     )

    transform = transforms.Compose(
    [
        
        # transforms.RandomVerticalFlip(0.45),
        # transforms.RandomHorizontalFlip(0.45),
        transforms.RandomRotation((0,5)),
        #transforms.v2.RandomResize((200, 250)), v2.RandomResize
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
        transforms.ColorJitter(brightness= (0.5, 1.5) , contrast=(0, 1.0)),
        transforms.Pad(20),
        transforms.Resize((224,224), antialias=True), 
        transforms.ToTensor(),
    ]
)
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

  
    
    
  
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        cfg.training.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    
    # with open("/kaggle/working/weights.json", "r") as f:
    #     weights = json.load(f)
  
    # sampler = WeightedRandomSampler(
    #     weights=weights,
    #     num_samples=len(weights),
    #     replacement=False
    # )
    #del data
    # Create train_loader and eval_loader here
    #if cfg.tokenizer.name is not None:
    train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.train_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast,
            collate_fn=collate_fn2)#, sampler= sampler)
    
    eval_loader = get_loader2(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn2)


    # Prepare everything using our accelerator
    (
        model,
        optimizer,
        train_loader,
        eval_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )
    
    #n_batches = len(train_loader)
    if not os.path.exists(cfg.output_dir):
        os.mkdir(cfg.output_dir)

    #print(device, model.device)
    custom_loss = CustomLoss() # use both cosine and euclidean similarities.
    #custom_bce_loss = CustomBCELoss()
    
    # for name , each in model.named_parameters():
    #     if torch.any(torch.isnan(each)):
    #         print(name, " layer has nan values in it..")
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps)) #, disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = epoch if epoch is not None else 0
    best_metric = None #loss
    #best_metric_checkpoint = None
    
    
    for epoch in range(starting_epoch, cfg.training.num_epochs):        
        #model.train()
        # if cfg.tracking:
        #     total_loss = 0
        train_losses = 0.0
        #check = True
        start_time = time()
        for step, (encoded_images,indication_prompt, true_stop_probs, reports) in enumerate(train_loader): #labels            
            #print("")
            # if step % 500 == 0:
            #     print("\nIndication prompt device: ", indication_prompt.device)
            #     print("step: ", step, encoded_images.shape, indication_prompt.shape, true_stop_probs.shape, reports.shape)
            
            # if torch.any(torch.isnan(encoded_images)):
            #     print("Raw images are nan..")
        
            loss = 0            
            n_sentences  = reports.shape[1]
            # print('Length of sentences: ', n_sentences)
            # print("Inputs shape: ", encoded_images.shape, indication_prompt.shape, labels.shape, true_stop_probs.shape, reports.shape)
            # print("True probs: ",true_stop_probs)
            
            encoder_pad_mask = create_padding_mask(indication_prompt).to(device)
            #print("Mem shape: ", indication_prompt.shape, "mask shape: ", encoder_pad_mask.shape)
            #encoder_causal_mask = src_mask(indication_prompt.shape[1])
           
            encoded_images  = model.encoder(encoded_images)#.type(torch.cuda.HalfTensor))
            
            # for name , each in model.encoder.named_parameters():
            #     if torch.any(torch.isnan(each)):
            #         print("After Image processing: ", name, " layer has nan values in it..")
                    
            # if torch.any(torch.isinf(encoded_images)) or torch.any(torch.isnan(encoded_images)):
            #     print("Encoded Images is nan")
            
            # if torch.any(torch.isinf(tags)) or torch.any(torch.isnan(tags)):
            #     print("Tags are nan")
            #print("Encoded Images: ", encoded_images.shape)
            bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
            
            # if model.co_attention:
            #     semantic_features = model.semantic_features_extractor(tags)
                
            #if model.history_encoder is not None:
            indication_prompt = model.decoder.embed_layer(indication_prompt)
            # if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
            #     print("Encoding by decoder embedding layer is nan")
                
            indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask.type(indication_prompt.dtype))
                # if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
                #     print("Encoding by History encoder is nan")  

            # # compute mask, confirm the first part.
            # mem_mask = torch.cat([torch.zeros((encoded_images.shape[0], encoded_images.shape[1])), encoder_pad_mask], dim=-1)
            
            #Initialize states
            if len(encoded_images.shape) > 3:
                encoded_images = encoded_images.reshape(bs, n_channels, -1)
                
            #print("lstm hidden state and cell state: ", model.sent_lstm.num_layers)
            
            prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt)
            #lstm_init = True
            #print(hn.shape, cn.shape)
            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            # if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
            #         print("Initial Prev hidden, hn, cn is nan")
                    
            for i in range(n_sentences):
                #print(f"This loop is for the number {i} sentence.")
                
                # for name , each in model.named_parameters():
                #     if torch.any(torch.isnan(each)):
                #         print(name, " layer has nan values in it..")
                    
                # if model.co_attention:
                #     if model.history_encoder is not None:
                #         #print("Encoded Images before attentino: ", encoded_images.shape)
                #         context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, indication_prompt)
                #     else:
                #         context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                # else:
                
                
                # Attend to encoded_images
                #if model.history_encoder is not None:
                context_vector, att_wts = model.attention(prev_hidden, encoded_images)                          
                # else:
                #     context_vector, att_wts = model.attention(prev_hidden, encoded_images)

                # if torch.any(torch.isinf(context_vector)) or torch.any(torch.isnan(context_vector)):
                #     print("Context Vector is nan")
                    
                # Generate Topic Vector
                #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
                prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
                #lstm_init= False
                # if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
                #     print("New prev hidden is nan")
                    
                # if torch.any(torch.isinf(hn)) or torch.any(torch.isnan(hn)):
                #     print("hn , cn are nan")
                    
                # if torch.any(torch.isinf(pred_stop_probs)) or torch.any(torch.isnan(pred_stop_probs)):
                #     print("Pred_stop_probs is nan")
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                #print("Target indices: ", tgt)
                #print('Target mask: ', tgt.shape)
                #padding_mask = create_padding_mask(tgt).to(device).type(indication_prompt.dtype)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1]).to(device).type(indication_prompt.dtype)
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None
                # if torch.any(torch.isinf(memory)) or torch.any(torch.isnan(memory)):
                #     print("Memory is affected contains nan values..")
                    
                # if torch.any(torch.isnan(encoder_pad_mask)) or  torch.any(torch.isnan(tgt_mask)):#torch.any(torch.isnan(padding_mask)) or
                #     print("Padding mask contains nan values...")
                # Compute attention for visual_features, encoded_prompt
                #memory = model.prompt_attention(memory, indication_prompt, key_padding_mask=mem_mask, residual_connection=True)
                # print(memory.shape, indication_prompt.shape, tgt.shape, prev_hidden.shape)
                # print(encoder_pad_mask.shape)
                output , final_decoder_layer_output = model.decoder(tgt, prev_hidden, (indication_prompt, memory), tgt_key_padding_mask= None,
                                    memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
                                        tgt_is_causal=False)  # [batch_size, seq_len - 1, d_model] 
                
                # if torch.any(torch.isinf(output)) or torch.any(torch.isnan(output)):
                #     print("Output is affected...")
                    
                # print("output shape: ", output.shape, reports[:, i, 1:].shape)
                # print("stop prob shape: ", pred_stop_probs.shape, true_stop_probs[:, 0].shape)
                
                loss += custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:],
                                    pred_stop_probs, output, prev_hidden, final_decoder_layer_output)  # Ignore <sos> token
                

            #loss += custom_bce_loss( tags, labels)
            # print("Loss: ", loss)
            # print("Loss shape: ",loss.shape)
            train_losses += loss.item()
    
            train_loss = train_losses/ (step + 1)
                      
            # We keep track of the loss at each epoch
            # if cfg.tracking:
            #     total_loss += loss.detach().float()
                
            # print("Loss before division by gradient accumulation: ", loss)
            loss = loss / cfg.training.gradient_accumulation_steps
            
            #print("Loss before backward: ", loss)
            accelerator.backward(loss)
            
            # #continue
            
            if step  % cfg.training.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                #completed_steps += 1
                
    #         # continue
            
            if step % cfg.training.eval_every == 0:
                model.eval()   
                print("\nEvaluating model...")
                eval_loss, eval_bce_loss, perplexity, sim_loss = evaluate(model, accelerator, eval_loader, custom_loss) #custom_bce_loss
                logger.info(f"\nEpoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}\n \
                    stop_loss {eval_bce_loss} similarity loss: {sim_loss} total_eval_loss {eval_loss + eval_bce_loss + sim_loss }" ) #label_loss: {label_loss} 
                
                model.train()
                #train_losses = []
                
                # Tracks the best checkpoint and best metric
                mean_loss = (train_loss + (eval_loss+eval_bce_loss +sim_loss))/2
                loss_diff = train_loss - (eval_loss+eval_bce_loss +sim_loss)
                
                if (best_metric is None or (best_metric > mean_loss and loss_diff > -0.65)):
                    best_metric = mean_loss
                    best_metric_checkpoint = os.path.join(cfg.output_dir, str(epoch))
                    #logger.info(f"New best metric: {best_metric} at epoch {epoch}")
                    logger.info(f"Saving model with best metric: Eval loss {best_metric}...")

                    epoch_dir = f"model_with_best_eval"
                    if cfg.output_dir is not None:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)            

                        
                            
                        output_dir = os.path.join(cfg.output_dir, epoch_dir)
                        if not os.path.exists(output_dir):
                            os.mkdir(output_dir)
                            
                        save_model(unwrapped_model, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, step=step, loss=loss, path=output_dir)
                        # unwrapped_model.save_pretrained(
                        #     output_dir,
                        #     is_main_process=accelerator.is_main_process,
                        #     save_function=accelerator.save,
                        #     state_dict=accelerator.get_state_dict(model),
                        # )
                
            #print("Back to training...")        
            if step % cfg.training.save_every == 0:                 
                epoch_dir = f"epoch_most_recent"
                
                logger.info(f"Saving most recent model in {epoch_dir}..")
                if cfg.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)            
                    output_dir = os.path.join(os.path.abspath(cfg.output_dir), epoch_dir)
                    
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                            
                    save_model(unwrapped_model, optimizer= optimizer, lr_scheduler=lr_scheduler, epoch=epoch, step=step, loss= loss, path =output_dir)
                    # unwrapped_model.save_pretrained(
                    #     output_dir,
                    #     is_man_process=accelerator.is_main_process,
                    #     save_function=accelerator.save,
                    #     state_dict=accelerator.get_state_dict(model),
                    # )
            #logger.info("Returning back to the next step...")
            if completed_steps >= cfg.training.max_train_steps:
                break
            
        eval_loss, eval_bce_loss, perplexity, sim_loss = evaluate(model,accelerator,eval_loader, custom_loss) #, custom_bce_loss
        model.train()
        logger.info(f"\nEpoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}\n \
            stop_loss {eval_bce_loss} similarity loss: {sim_loss} total_eval_loss {eval_loss + eval_bce_loss + sim_loss }" ) #label_loss: {label_loss} 
        
    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        output_dir = os.path.join(os.path.abspath(cfg.output_dir), "final")
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
                            
        save_model(unwrapped_model, path= output_dir)
        # unwrapped_model.save_pretrained(
        #     os.path.join(os.path.abspath(cfg.output_dir),"final"),
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(model),
        #)
        
    return




if __name__ == "__main__":
    cfg = OmegaConf.load("/kaggle/working/MIR/conf/config2.yaml") #
    # try:
    #     with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    train(cfg)
    #         print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # except:
    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #cfg = OmegaConf.load("./conf/config.yaml")
    # #train(cfg)
    # model = load_model(cfg)
    #optimizer = torch.optim.AdamW(model.parameters(), lr= cfg.training.learning_rate)
    
    # if not os.path.exists(os.path.join(os.path.abspath(os.curdir),"final") ):
    #     os.mkdir(os.path.join(os.path.abspath(os.curdir),"final") )
    #save_model(model, optimizer = optimizer,path = os.path.join(os.path.abspath(os.curdir),"final") )
    # model, optimizer, epoch, loss = load_model(cfg)
    # print(model, optimizer, epoch, loss)
    
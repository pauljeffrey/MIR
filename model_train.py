# my_script.py

from torch import nn
from utils import *
from loss import CustomLoss
from dataset import *
#from build_vocab import load_vocab
from full_model import * 

from torchvision import transforms

from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
import logging
import math
import os
import torch

from tqdm.auto import tqdm
from transformers import (
    get_scheduler,
)
torch.manual_seed(42)
from utils import src_mask, create_padding_mask
from new_train import load, load_model, save_model
import torch
import gc

def evaluate(model, eval_loader, custom_loss): #, bce_loss
    model.eval()
    device="cuda"
    with torch.no_grad():
        eval_losses = []
        eval_stop_losses = []
        #eval_bce_losses = []
        print("In evaluation function ....")
        for _ , (encoded_images, indication_prompt, true_stop_probs, reports) in enumerate(eval_loader):   #labels,      
            #n_sentences  = reports.shape[1]
            encoded_images = encoded_images.to(device)
            indication_prompt = torch.tensor(indication_prompt).type(torch.LongTensor).to(device)
            true_stop_probs = torch.tensor(true_stop_probs).type(torch.LongTensor).to(device)
            reports = torch.tensor(reports).type(torch.LongTensor).to(device)
            
            n_sentences  = reports.shape[1]
            
            encoder_pad_mask = create_padding_mask(indication_prompt).to(device)
            encoded_images  = model.encoder(encoded_images)#.type(torch.cuda.HalfTensor)) # , tags
            
            # if torch.any(torch.isinf(encoded_images)) or torch.any(torch.isnan(encoded_images)):
            #     print("Encoded Images is nan")
            bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
                
            if model.history_encoder is not None:
                indication_prompt = model.decoder.embed_layer(indication_prompt)
                    
                indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask.type(indication_prompt.dtype))
                # if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
                #     print("Encoding by History encoder is nan")  
            
            #Initialize states
            if len(encoded_images.shape) > 3:
                encoded_images = encoded_images.reshape(bs, n_channels, -1)
                
            #print("lstm hidden state and cell state: ", model.sent_lstm.num_layers)
            
            prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt)
            # if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
            #         print("Initial Prev hidden, hn, cn is nan")
                    
            for i in range(n_sentences):
                ##print(f"This loop is for the number {i} sentence.")
                
                # for name , each in model.named_parameters():
                #     if torch.any(torch.isnan(each)):
                #         print(name, " layer has nan values in it..")
                
                # Attend to encoded_images
                if model.history_encoder is not None:
                    context_vector, att_wts = model.attention(prev_hidden, encoded_images, indication_prompt)                          
                else:
                    context_vector, att_wts = model.attention(prev_hidden, encoded_images)
                prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
                #lstm_init= False
                # if torch.any(torch.isinf(prev_hidden)) or torch.any(torch.isnan(prev_hidden)):
                #     print("New prev hidden is nan")
                    
                # if torch.any(torch.isinf(hn)) or torch.any(torch.isnan(hn)):
                #     print("hn , cn are nan")
                    
                # if torch.any(torch.isinf(pred_stop_probs)) or torch.any(torch.isnan(pred_stop_probs)):
                #     print("Pred_stop_probs is nan")
                # Decode reports
                tgt = reports[:,i, :-1] 
                tgt_mask = src_mask(tgt.shape[1]).to(device).type(indication_prompt.dtype)
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None
                
                output = model.decoder(tgt, prev_hidden, (indication_prompt, memory), tgt_key_padding_mask=None,
                                    memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
                                        tgt_is_causal=False)  #
                
                #loss += custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:], pred_stop_probs,  output, eval=True)  # Ignore <sos> token
                stop_loss, sparse_loss = custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:],pred_stop_probs,  output, eval=True)  # Ignore <sos> token
               
                eval_stop_losses.append(stop_loss.item())
                eval_losses.append(sparse_loss.item()) 
                
        try:
            eval_loss = sum(eval_losses)/ len(eval_losses)
            eval_stop_loss = sum(eval_stop_losses)/len(eval_stop_losses)
            #eval_bce_loss = torch.mean(torch.cat(eval_bce_losses))
            perplexity = math.exp(eval_loss)
            
            
        except OverflowError:
            perplexity = float("inf")
                    
    return eval_loss , eval_stop_loss, perplexity #eval_bce_loss

@profile
def train(cfg: DictConfig):
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = "cuda"
    # logger.info(OmegaConf.to_yaml(cfg))
    
    if not cfg.model.from_checkpoint:
        model = load_model(cfg, device= device)
        epoch = None
        loss = None
        # Optimizer
        # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
        optimizer_cls = (
            torch.optim.AdamW 
        )
        optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)

    else:
        model, optimizer, epoch, loss = load_model(cfg)
        
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )

    transform = transforms.Compose(
    [
        # transforms.RandomRotation((0,5)),
        # #transforms.v2.RandomResize((200, 250)), v2.RandomResize
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
        # transforms.ColorJitter(brightness= (0.5, 1.5) , contrast=(0, 1.0)),
        transforms.Pad(20),
        transforms.Resize((224,224), antialias=True), 
        transforms.ToTensor(),
    ]
)
 
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps)) #, disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = epoch if epoch is not None else 0
    best_metric = loss
    best_metric_checkpoint = None
    
    
    train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.train_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast,
            collate_fn=collate_fn2)#, sampler= sampler)
    
    eval_loader = get_loader2(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn2)
  

    #n_batches = len(train_loader)
    if not os.path.exists(cfg.output_dir):
        os.mkdir(cfg.output_dir)

    #print(device, model.device)
    custom_loss = CustomLoss()
   
    for epoch in range(starting_epoch, cfg.training.num_epochs):        
        model.train()
        # if cfg.tracking:
        #     total_loss = 0
        train_losses = []
        #check = True
        for step, (encoded_images,indication_prompt, true_stop_probs, reports) in enumerate(train_loader): #labels,
            # indication_prompt = torch.tensor(indication_prompt).type(torch.LongTensor).to(device)
            # true_stop_probs = torch.tensor(true_stop_probs).type(torch.LongTensor).to(device)
            # reports = torch.tensor(reports).type(torch.LongTensor).to(device)
            if step % 500 == 0:
                print(step, encoded_images.shape, indication_prompt.shape, true_stop_probs.shape, reports.shape)
            
            loss = 0      
            gc.collect() 
            if step > 4500:
                break      
            # n_sentences  = reports.shape[1]
            
    #         encoder_pad_mask = create_padding_mask(indication_prompt).to(device)
           
    #         encoded_images  = model.encoder(encoded_images.to(device))
    #         bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
            
    #         # if model.co_attention:
    #         #     semantic_features = model.semantic_features_extractor(tags)
                
    #         if model.history_encoder is not None:
    #             indication_prompt = model.decoder.embed_layer(indication_prompt.to(device))
    #             # if torch.any(torch.isinf(indication_prompt)) or torch.any(torch.isnan(indication_prompt)):
    #             #     print("Encoding by decoder embedding layer is nan")
                    
    #             indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask.type(indication_prompt.dtype))
            
    #         #Initialize states
    #         if len(encoded_images.shape) > 3:
    #             encoded_images = encoded_images.reshape(bs, n_channels, -1)
                
    #         #print("lstm hidden state and cell state: ", model.sent_lstm.num_layers)
            
    #         prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt)
                    
    #         for i in range(n_sentences):                
    #             # Attend to encoded_images
    #             if model.history_encoder is not None:
    #                 context_vector, att_wts = model.attention(prev_hidden, encoded_images, indication_prompt)                          
    #             else:
    #                 context_vector, att_wts = model.attention(prev_hidden, encoded_images)

    #             # Generate Topic Vector
    #             #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
    #             prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
              
    #             # Decode reports
    #             tgt = reports[:,i, :-1].to(device)  # Remove last token from reports
               
    #             tgt_mask = src_mask(tgt.shape[1]).to(device).type(indication_prompt.dtype)
                
    #             memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
               
    #             output = model.decoder(tgt, prev_hidden, (indication_prompt, memory), tgt_key_padding_mask= None,
    #                                 memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
    #                                     tgt_is_causal=False)  # [batch_size, seq_len - 1, d_model] 
            
                
    #             loss += custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:].to(device), pred_stop_probs,  output)  # Ignore <sos> token

    #         #loss += custom_bce_loss( tags, labels)
            
    #         train_losses.append(
    #                 loss.item()
    #             )
    
    #         train_loss = sum(train_losses) / len(train_losses) 
    #         loss = loss / cfg.training.gradient_accumulation_steps
    #         loss.backward()
            
    #         #continue
            
    #         if step  % cfg.training.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             progress_bar.update(1)
    #             completed_steps += 1
                
    # #         # continue
            
    #         if step % cfg.training.eval_every == 0:
    #             model.eval()   
    #             #print("\nEvaluating model...")
    #             eval_loss, eval_bce_loss, perplexity = evaluate(model, eval_loader, custom_loss) #custom_bce_los
                
    #             print(f"\nEpoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
    #                 stop_loss {eval_bce_loss} total_eval_loss {eval_loss + eval_bce_loss}" ) #label_loss: {label_loss} 
                
    #             model.train()
    #             #train_losses = []
                
    #             # Tracks the best checkpoint and best metric
    #             mean_loss = (train_loss + eval_loss)/2
    #             loss_diff = train_loss - eval_loss
                
    #             if (best_metric is None or (best_metric > mean_loss and loss_diff > -0.65)):
    #                 best_metric = mean_loss
    #                 best_metric_checkpoint = os.path.join(cfg.output_dir, str(epoch))
                
    #                 epoch_dir = f"model_with_best_eval"
    #                 if cfg.output_dir is not None:                 
                            
    #                     output_dir = os.path.join(cfg.output_dir, epoch_dir)
    #                     if not os.path.exists(output_dir):
    #                         os.mkdir(output_dir)
                            
    #                     save_model(model, optimizer=optimizer, epoch=epoch, loss=loss, path=output_dir)
                       
    #         #print("Back to training...")        
    #         if step % cfg.training.save_every == 0:                 
    #             epoch_dir = f"epoch_most_recent"
    #             if cfg.output_dir is not None:           
    #                 output_dir = os.path.join(os.path.abspath(cfg.output_dir), epoch_dir)
                    
    #                 if not os.path.exists(output_dir):
    #                     os.mkdir(output_dir)
                            
    #                 save_model(model, optimizer= optimizer, epoch=epoch, loss= loss, path =output_dir)
                
             

            # if completed_steps >= cfg.training.max_train_steps:
            #     break
            
    #     eval_loss, eval_bce_loss, perplexity = evaluate(model,eval_loader, custom_loss) #, custom_bce_loss
    #     model.train()
    #     print(f"\nEpoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
    #                 stop_loss {eval_bce_loss} total_eval_loss {eval_loss + eval_bce_loss}" )
        
    # print('Saving the model using the best weights checkpoint in the current output directory')
    # if cfg.output_dir is not None:
    #     output_dir = os.path.join(os.path.abspath(cfg.output_dir), "final")
        
    #     if not os.path.exists(output_dir):
    #         os.mkdir(output_dir)
                            
    #     save_model(model, path= output_dir)
       
    return


if __name__ == "__main__":
    cfg = OmegaConf.load("/kaggle/working/MIR/conf/config.yaml") #
    train(cfg)
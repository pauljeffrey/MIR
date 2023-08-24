
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
#from accelerate.utils import DummyOptim, DummyScheduler, set_seed

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



def save_model(model , save_dir):
    if not os.path.exists("save_dir"):
        os.mkdir(save_dir)
        
    path = os.path.abspath(save_dir)
    
    # Save each module
    if model.encoder is not None:
        torch.save(model.encoder.state_dict(), os.path.join(path, 'encoder.pt'))
    
    if model.history_encoder is not None:
        torch.save(model.history_encoder.state_dict(), os.path.join(path, "prompt_encoder.pt"))
        
    if model.semantic_features_extractor is not None:
        torch.save(model.semantic_features_extractor.state_dict(), os.path.join(path, "semantic_features_extractor.pt"))
        
    torch.save(model.attention.state_dict(), os.path.join(path, "attention.pt"))
    torch.save(model.sent_lstm.state_dict(), os.path.join(path, "sent_lstm.pt"))
    torch.save(model.decoder.state_dict(), os.path.join(path, "decoder.pt"))

    return

def load(model, save_dir):
    successful_loads = 0
    if model.encoder is not None and os.path.exists(os.path.join(save_dir, "encoder.pt")):
        model.encoder.load_state_dict(torch.load(os.path.join(save_dir, "encoder.pt")))
        successful_loads += 1
        
    if model.history_encoder is not None and os.path.exists(os.path.join(save_dir, "prompt_encoder.pt")):
        model.history_encoder.load_state_dict(torch.load(os.path.join(save_dir, "prompt_encoder.pt")))
        successful_loads += 1
        
    if model.semantic_features_extractor is not None and os.path.exists(os.path.join(save_dir, "semantic_features_extractor.pt")):
        model.semantic_features_extractor.load_state_dict(torch.load(os.path.join(save_dir, "semantic_features_extractor.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "attention.pt")):
        model.attention.load_state_dict(torch.load(os.path.join(save_dir, "attention.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "sent_lstm.pt")):
        model.sent_lstm.load_state_dict(torch.load(os.path.join(save_dir, "sent_lstm.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "decoder.pt")):
        model.decoder.load_state_dict(torch.load(os.path.join(save_dir, "decoder.pt")))
        successful_loads += 1
        
    print(f"Successfully loaded {successful_loads} sub models.")
    
    return model

#@hydra.main(version_base=None, config_path="conf", config_name="config")
def load_model(cfg: DictConfig):
    model_params = cfg.architecture
    model = MedicalReportGenerator(**model_params)
    if cfg.model.from_trained:
        path = os.path.join(os.path.abspath(cfg.output_dir + cfg.load_dir)) # Assumes that all state_dicts are stored in the same directory.
        print(f"loading all sub model weights from {path}...")
        model = load(model, cfg.load_dir)
    return model


def evaluate(model, accelerator, eval_loader, custom_loss, bce_loss):
    model.eval()
    loss = 0
    with torch.no_grad():
        eval_losses = []
        eval_stop_losses = []
        eval_bce_losses = []
        for _ , (encoded_images,indication_prompt, labels, true_stop_probs, reports) in enumerate(eval_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)

           
            loss = 0            
            n_sentences  = reports.shape[1]
            
            encoder_pad_mask = create_padding_mask(indication_prompt).to(accelerator.device)
            #encoder_causal_mask = src_mask(indication_prompt.shape[1])
            
            encoded_images , tags = model.encoder(encoded_images)
            #print("Encoded Images: ", encoded_images.shape)
            bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
            
            if model.co_attention:
                semantic_features = model.semantic_features_extractor(tags)
                
            if model.history_encoder is not None:
                indication_prompt = model.decoder.embed_layer(indication_prompt)
                indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask)
                    

            # # compute mask, confirm the first part.
            # mem_mask = torch.cat([torch.zeros((encoded_images.shape[0], encoded_images.shape[1])), encoder_pad_mask], dim=-1)
            
            #Initialize states
            if len(encoded_images.shape) > 3:
                encoded_images = encoded_images.reshape(bs, n_channels, -1)
                
            #print(model.sent_lstm.num_layers)
            
            prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt)
            #lstm_init = True
            #print(hn.shape, cn.shape)
            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        #print("Encoded Images before attentino: ", encoded_images.shape)
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, indication_prompt)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, indication_prompt)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)

                # Generate Topic Vector
                #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
                prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
                #lstm_init= False
                
                # Decode reports
                #print(reports.shape)
                tgt = reports[:,i, :-1]   # Remove last token from reports
                padding_mask = create_padding_mask(tgt)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1]).to(accelerator.device)
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None
                
                # Compute attention for visual_features, encoded_prompt
                #memory = model.prompt_attention(memory, indication_prompt, key_padding_mask=mem_mask, residual_connection=True)
                # print(memory.shape, indication_prompt.shape, tgt.shape, prev_hidden.shape)
                # print(encoder_pad_mask.shape)
                output = model.decoder(tgt, prev_hidden, (indication_prompt,memory),tgt_key_padding_mask=padding_mask,
                                    memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
                                        tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
                
                stop_loss, sparse_loss = custom_loss(true_stop_probs[:,i], reports[:, i, 1:],pred_stop_probs,  output, eval=True)  # Ignore <sos> token
                
                eval_stop_losses.append(accelerator.gather(stop_loss).detach().cpu())
                eval_losses.append(accelerator.gather(sparse_loss).detach().cpu())
                

            binary_loss = bce_loss(tags, labels)
            eval_bce_losses.append(accelerator.gather(binary_loss).detach().cpu())
            

        try:
            eval_loss = torch.mean(torch.cat(eval_losses))
            eval_stop_loss = torch.mean(torch.cat(eval_stop_losses))
            eval_bce_loss = torch.mean(torch.cat(eval_bce_losses))
            perplexity = math.exp(eval_loss)
            
            
        except OverflowError:
            perplexity = float("inf")
                    
    return eval_loss , eval_stop_loss, perplexity, eval_bce_loss




#@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
    accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin =deepspeed_plugin)
    
    accelerator.wait_for_everyone()
    device= accelerator.device
   

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    model = load_model(cfg)

    # Optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW #Adafactor #
        # if accelerator.state.deepspeed_plugin is None
        # or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        # else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)


    # if (
    #     accelerator.state.deepspeed_plugin is None
    #     or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    # ):
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )
    # else:
    #     lr_scheduler = DummyScheduler(
    #         optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
    #     )

    transform = transforms.Compose(
    [
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
    ]
)
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

  
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    #starting_epoch = 0
    best_metric = None
    #best_metric_checkpoint = None
    
    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        cfg.training.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    
    # Create train_loader and eval_loader here
    #if cfg.tokenizer.name is not None:
    train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.train_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn2)
    
    eval_loader = get_loader2(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, 
            tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.training.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
            n_max=cfg.dataset.tokens.n_max, encoder_n_max=cfg.dataset.tokens.encoder_n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn2)
    # else:
    #     vocabulary1 = load_vocab(cfg.vocabs.name1)
    #     train_loader = get_loader(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json,cfg.dataset.train.history_json, cfg.dataset.train.file_list,
    #            vocabulary = vocabulary1, vocabulary2=vocabulary2, transform= transform, batch_size = cfg.dataset.train_batch_size, s_max= cfg.dataset.tokens.s_max,
    #            n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn)
    
    #     eval_loader = get_loader(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, cfg.dataset.eval.history_json, cfg.dataset.eval.file_list,
    #            vocabulary = vocabulary1, vocabulary2=vocabulary2, transform= transform, batch_size = cfg.dataset.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
    #            n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn)


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
    

    device = accelerator.device
    print(device, model.device)
    custom_loss = CustomLoss()
    custom_bce_loss = CustomBCELoss()

    for epoch in range(cfg.training.num_epochs):
        model.train()
                
        # if cfg.tracking:
        #     total_loss = 0
        train_losses = []
        
        for step, (encoded_images,indication_prompt, labels,true_stop_probs, reports) in enumerate(train_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)

            loss = 0            
            n_sentences  = reports.shape[1]
            
            encoder_pad_mask = create_padding_mask(indication_prompt).to(device)
            print("Mem shape: ", indication_prompt.shape, "mask shape: ", encoder_pad_mask.shape)
            #encoder_causal_mask = src_mask(indication_prompt.shape[1])
            
            encoded_images , tags = model.encoder(encoded_images.type(torch.cuda.HalfTensor))
            #print("Encoded Images: ", encoded_images.shape)
            bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
            
            if model.co_attention:
                semantic_features = model.semantic_features_extractor(tags)
                
            if model.history_encoder is not None:
                indication_prompt = model.decoder.embed_layer(indication_prompt)
                indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask.type(indication_prompt.dtype))
                    

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
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        #print("Encoded Images before attentino: ", encoded_images.shape)
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, indication_prompt)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, indication_prompt)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)

                # Generate Topic Vector
                #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
                prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
                #lstm_init= False
                
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                #print('Target mask: ', tgt.shape)
                padding_mask = create_padding_mask(tgt).to(device).type(indication_prompt.dtype)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1]).to(device).type(indication_prompt.dtype)
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None
                
                # Compute attention for visual_features, encoded_prompt
                #memory = model.prompt_attention(memory, indication_prompt, key_padding_mask=mem_mask, residual_connection=True)
                # print(memory.shape, indication_prompt.shape, tgt.shape, prev_hidden.shape)
                # print(encoder_pad_mask.shape)
                output = model.decoder(tgt, prev_hidden, (indication_prompt,memory),tgt_key_padding_mask=padding_mask,
                                    memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
                                        tgt_is_causal=False)  # [batch_size, seq_len - 1, d_model]
                
                loss += custom_loss(true_stop_probs[:,i].type(indication_prompt.dtype), reports[:, i, 1:],pred_stop_probs,  output)  # Ignore <sos> token

            loss += custom_bce_loss( tags, labels)
            
            train_losses.append(
                    accelerator.gather(loss.repeat(cfg.training.train_batch_size))
                )
    
            train_loss = torch.mean(torch.cat(train_losses))  
                      
            # We keep track of the loss at each epoch
            if cfg.tracking:
                total_loss += loss.detach().float()
                
            loss = loss / cfg.training.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step  % cfg.training.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                
            if step % cfg.training.eval_every == 0:
                eval_loss, eval_bce_loss, perplexity, label_loss = evaluate(model, accelerator, eval_loader, custom_loss, custom_bce_loss)
                logger.info(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
                    stop_loss {eval_bce_loss} label_loss: {label_loss} total_eval_loss {eval_loss + eval_bce_loss + label_loss}" )
                
                print(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
                    stop_loss {eval_bce_loss} label_loss: {label_loss} total_eval_loss {eval_loss + eval_bce_loss + label_loss}" )
                model.train()
                
                # Tracks the best checkpoint and best metric
                mean_loss = (train_loss + eval_loss)/2
                loss_diff = train_loss - eval_loss
                
                if (best_metric is None or (best_metric > mean_loss and loss_diff > -0.65)):
                    best_metric = mean_loss
                    best_metric_checkpoint = os.path.join(cfg.output_dir, str(epoch))
                    logger.info(f"New best metric: {best_metric} at epoch {epoch}")
                    logger.info(f"Saving model with best metric: Eval loss {best_metric}...")

                    epoch_dir = "model_with_best_eval"
                    if cfg.output_dir is not None:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)            

                        output_dir = os.path.join(cfg.output_dir, epoch_dir)

                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        
            if step % cfg.training.save_every == 0:                 
                epoch_dir = f"epoch_{epoch}_most_recent"
                
                logger.info(f"Saving model in {epoch_dir}..")
                if cfg.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)            
                    output_dir = os.path.join(os.path.abspath(cfg.output_dir), epoch_dir)
                    
                    save_model(unwrapped_model, output_dir)
                    # unwrapped_model.save_pretrained(
                    #     output_dir,
                    #     is_man_process=accelerator.is_main_process,
                    #     save_function=accelerator.save,
                    #     state_dict=accelerator.get_state_dict(model),
                    # )

            if completed_steps >= cfg.training.max_train_steps:
                break
            
        eval_loss, eval_bce_loss, perplexity, label_loss = evaluate(model,accelerator,eval_loader, custom_loss, custom_bce_loss)
        model.train()
        logger.info(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
                    stop_loss {eval_bce_loss} label_loss: {label_loss} total_eval_loss {eval_loss + eval_bce_loss + label_loss}" )
        print(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} sparse_loss: {eval_loss}  \
                    stop_loss {eval_bce_loss} label_loss: {label_loss} total_eval_loss {eval_loss + eval_bce_loss + label_loss}" )
        
    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        save_model(unwrapped_model, os.path.join(os.path.abspath(cfg.output_dir), "final"))
        # unwrapped_model.save_pretrained(
        #     os.path.join(os.path.abspath(cfg.output_dir),"final"),
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(model),
        # )
        
    return




if __name__ == "__main__":
    cfg = OmegaConf.load("/kaggle/working/MIR/conf/config.yaml")
    train(cfg)
    # cfg = OmegaConf.load("./conf/config.yaml")
    # #train(cfg)
    # model = load_model(cfg)
    #save_model(model, "pretrained")
    # images = torch.randn((256,3,224,224))
    # prompt = torch.randint(0, 5000, (256, 30))
    # true_labels = torch.randint(0,1, (256, 156))
    # real_stop_probs = torch.randint(-1,1,(256, 20 ))
    # true_reports = torch.randint(0,5000, (256, 20, 80))
    
    # custom_bce_loss = CustomBCELoss()
    # custom_loss = CustomLoss()
    # batch_size = 8
    
    # for ind in range(0, 256, batch_size):
    #     encoded_images = images[ind: ind+batch_size] 
    #     indication_prompt = prompt[ind: ind+batch_size]
    #     labels = true_labels[ind: ind+batch_size]
    #     true_stop_probs = real_stop_probs[ind: ind+batch_size]
    #     reports = true_reports[ind: ind+batch_size]

        
    #     loss = 0            
    #     n_sentences  = reports.shape[1]
        
    #     encoder_pad_mask = create_padding_mask(indication_prompt)
    #     #encoder_causal_mask = src_mask(indication_prompt.shape[1])
        
    #     encoded_images , tags = model.encoder(encoded_images)
    #     #print("Encoded Images: ", encoded_images.shape)
    #     bs , n_channels = encoded_images.shape[0], encoded_images.shape[1]
        
    #     if model.co_attention:
    #         semantic_features = model.semantic_features_extractor(tags)
            
    #     if model.history_encoder is not None:
    #         indication_prompt = model.decoder.embed_layer(indication_prompt)
    #         indication_prompt = model.history_encoder(indication_prompt, mask=encoder_pad_mask)
                

    #     # # compute mask, confirm the first part.
    #     # mem_mask = torch.cat([torch.zeros((encoded_images.shape[0], encoded_images.shape[1])), encoder_pad_mask], dim=-1)
        
    #     #Initialize states
    #     if len(encoded_images.shape) > 3:
    #         encoded_images = encoded_images.reshape(bs, n_channels, -1)
            
    #     #print(model.sent_lstm.num_layers)
        
    #     prev_hidden, (hn, cn) = model.sent_lstm.init_state(encoded_images,indication_prompt)
    #     #lstm_init = True
    #     #print(hn.shape, cn.shape)
    #     #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
    #     for i in range(n_sentences):
            
    #         if model.co_attention:
    #             if model.history_encoder is not None:
    #                 #print("Encoded Images before attentino: ", encoded_images.shape)
    #                 context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, indication_prompt)
    #             else:
    #                 context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
    #         else:
    #             # Attend to encoded_images
    #             if model.history_encoder is not None:
    #                 context_vector, att_wts = model.attention(prev_hidden, encoded_images, indication_prompt)                          
    #             else:
    #                 context_vector, att_wts = model.attention(prev_hidden, encoded_images)

    #         # Generate Topic Vector
    #         #print("Context and hidden shape before entering lstm: ", context_vector.shape, prev_hidden.shape)
    #         prev_hidden, pred_stop_probs, (hn, cn) = model.sent_lstm(context_vector, prev_hidden, (hn, cn))  # [batch_size, d_model]
    #         #lstm_init= False
            
    #         # Decode reports
    #         tgt = reports[:,i, :-1]  # Remove last token from reports
    #         padding_mask = create_padding_mask(tgt)
    #         #causal_mask1 = create_causal_masks(inputs)
    #         tgt_mask = src_mask(tgt.shape[1])
            
    #         memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
    #         #memory_mask = None
            
    #         # Compute attention for visual_features, encoded_prompt
    #         #memory = model.prompt_attention(memory, indication_prompt, key_padding_mask=mem_mask, residual_connection=True)
    #         # print(memory.shape, indication_prompt.shape, tgt.shape, prev_hidden.shape)
    #         # print(encoder_pad_mask.shape)
    #         output = model.decoder(tgt, prev_hidden, (indication_prompt,memory),tgt_key_padding_mask=padding_mask,
    #                                memory_key_padding_mask=encoder_pad_mask, tgt_mask=tgt_mask,
    #                                 tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
            
    #         loss += custom_loss(true_stop_probs[:,i], reports[:, i, 1:],pred_stop_probs,  output)  # Ignore <sos> token

    #     loss += custom_bce_loss(labels, tags)
   
         
     

        
    #torch.save(model.state_dict(), "./pretrained")
    
    
   
        

    
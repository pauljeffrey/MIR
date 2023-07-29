
from torch import nn
from utils import *
from loss import *
from dataset import *
from build_vocab import load_vocab
from full_model import * 

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import AdamW,Adafactor, get_linear_schedule_with_warmup

from accelerate import Accelerator
from accelerate import Accelerator, DistributedType# ,DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import logging
import math
import os
import torch
import hydra
import torch

from tqdm.auto import tqdm
from transformers import (
    get_scheduler,
)


def load_model(cfg: DictConfig,):
    model_params = cfg.encoder + cfg.history_encoder + cfg.semantic_extractor + cfg.attention + cfg.sent_lstm + cfg.decoder
    model = MedicalReportGenerator(**model_params)
    if cfg.model.from_trained:
        path = os.path.join(os.path.abspath(cfg.output_dir + cfg.load_dir)) # Assumes that all state_dicts are stored in the same directory.
        model.load_state_dict(path)
    return model


def evaluate(model, accelerator, eval_loader, cross_entropy_loss):
    model.eval()
                
    with torch.no_grad():
        eval_losses = []
        for _ , (encoded_images, patient_history , reports, true_stop_probs) in enumerate(eval_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)
            loss = 0
            
            n_sentences  = reports.shape[1]
    
            if model.encoder is not None:    
                if model.co_attention:
                    encoded_images , tags = model.encoder(encoded_images)
                    semantic_features = model.semantic_features_extractor(tags)
                else:
                    encoded_images = model.encoder(encoded_images)
                    
            if model.history_encoder is not None:
                patient_history = model.history_encoder(patient_history)
            
            prev_hidden, prev_cell_state = model.lstm.init_state(encoded_images)
            lstm_init = True

            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, patient_history)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, patient_history)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)  

                # Generate Topic Vector
                prev_hidden, pred_stop_probs = model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)  # [batch_size, d_model]
                lstm_init = False
                
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                padding_mask = create_padding_mask(tgt)
                
                tgt_mask = src_mask(tgt.shape[1])
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None

                output = model.decoder(tgt, prev_hidden, memory,tgt_key_padding_mask=padding_mask, tgt_mask=tgt_mask,
                                    tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
                
                loss += cross_entropy_loss(true_stop_probs, reports[:, i, 1:], pred_stop_probs,  output)  # Ignore <sos> token
                
                gathered_loss= accelerator.gather(loss).detach().cpu()
                eval_losses.append(gathered_loss)

        losses = torch.cat(eval_losses)

        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
                    
    return eval_loss , perplexity


def evaluate_full(model, accelerator, eval_loader, cross_entropy_loss, bce_loss):
    model.eval()
    loss = 0
    with torch.no_grad():
        eval_losses = []
        eval_bce_losses = []
        for _ , (encoded_images,patient_history, labels, reports, true_stop_probs) in enumerate(eval_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)
            
            n_sentences  = reports.shape[1]
    
            if model.encoder:    
                if model.co_attention:
                    encoded_images , tags = model.encoder(encoded_images)
                    semantic_features = model.semantic_features_extractor(tags)
                else:
                    encoded_images = model.encoder(encoded_images)
                    
            if model.history_encoder is not None:
                patient_history = model.history_encoder(patient_history)
                    
            prev_hidden, prev_cell_state = model.lstm.init_state(encoded_images)
            lstm_init = True

            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, patient_history)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, patient_history)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)   

                # Generate Topic Vector
                prev_hidden, pred_stop_probs = model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)  # [batch_size, d_model]
                lstm_init = False
                
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                padding_mask = create_padding_mask(tgt)
                
                tgt_mask = src_mask(tgt.shape[1])
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None

                output = model.decoder(tgt, prev_hidden, memory,tgt_key_padding_mask=padding_mask, tgt_mask=tgt_mask,
                                    tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
                
                loss += cross_entropy_loss(true_stop_probs, reports[:, i, 1:], pred_stop_probs,  output)  # Ignore <sos> token
                
                gathered_loss= accelerator.gather(loss).detach().cpu()
                eval_losses.append(gathered_loss)

            binary_loss = bce_loss(tags, labels)
            eval_bce_losses.append(accelerator.gather(binary_loss).detach().cpu())
            

        try:
            eval_loss = torch.mean(torch.cat(eval_losses))
            eval_bce_loss = torch.mean(torch.cat(eval_bce_losses))
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
                    
    return eval_loss , eval_bce_loss, perplexity


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_decoder(cfg: DictConfig):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator()
    
    accelerator.wait_for_everyone()
    device= accelerator.device
   

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    model = load_model(cfg)

    # Optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        Adafactor #torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)


    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
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
    if cfg.tokenizer.name is not None:
        train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, cfg.dataset.train.file_list,
                tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.dataset.train_batch_size, s_max= cfg.dataset.tokens.s_max,
                n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast)
        
        eval_loader = get_loader2(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, cfg.dataset.eval.file_list,
                tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.dataset.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
                n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast)
    else:
        vocabulary1 = load_vocab(cfg.vocabs.name1)
        vocabulary2 = load_vocab(cfg.vocabs.name2)
        train_loader = get_loader(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json,cfg.dataset.train.history_json, cfg.dataset.train.file_list,
               vocabulary = vocabulary1, vocabulary2= vocabulary2, transform= transform, batch_size = cfg.dataset.train_batch_size, s_max= cfg.dataset.tokens.s_max,
               n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn2)
    
        eval_loader = get_loader(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, cfg.dataset.eval.history_json, cfg.dataset.eval.file_list,
               vocabulary = vocabulary1, vocabulary2=vocabulary2, transform= transform, batch_size = cfg.dataset.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
               n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn2)


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
    custom_loss = CustomLoss()
    
    if model.encoder is not None:
        model.encoder.requires_grad = False

    for epoch in range(cfg.training.num_epochs):
        model.train()
                
        if cfg.tracking:
            total_loss = 0
        train_losses = []

        for step, (encoded_images,patient_history, reports, true_stop_probs) in enumerate(train_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)

            loss = 0            
            n_sentences  = reports.shape[1]
            
            if model.encoder:
                if model.co_attention:
                    encoded_images , tags = model.encoder(encoded_images)
                    semantic_features = model.semantic_features_extractor(tags)
                else:
                    encoded_images = model.encoder(encoded_images)
                    
            if model.history_encoder is not None:
                patient_history = model.history_encoder(patient_history)
                    
            prev_hidden, prev_cell_state = model.lstm.init_state(encoded_images)
            lstm_init = True

            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, patient_history)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, patient_history)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)
                        
                # Generate Topic Vector
                prev_hidden, pred_stop_probs = model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)  # [batch_size, d_model]
                lstm_init= False
                
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                padding_mask = create_padding_mask(tgt)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1])
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None

                output = model.decoder(tgt, prev_hidden, memory,tgt_key_padding_mask=padding_mask, tgt_mask=tgt_mask,
                                       tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
                
                loss += custom_loss(true_stop_probs, reports[:, i, 1:], pred_stop_probs,  output)  # Ignore <sos> token

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
                eval_loss, perplexity = evaluate(model, accelerator, eval_loader, custom_loss)
                logger.info(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} eval_loss:{eval_loss} perplexity: {perplexity}")
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
                    output_dir = os.path.join(cfg.output_dir, epoch_dir)
                    
                    unwrapped_model.save_pretrained(
                        output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model),
                    )

            if completed_steps >= cfg.training.max_train_steps:
                break
            
        eval_loss, perplexity = evaluate(model,accelerator,eval_loader, custom_loss)
        model.train()
        logger.info(f"epoch {epoch}: perplexity: {perplexity} train_loss: {train_loss} eval_loss: {eval_loss}")
 
        
    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            os.path.join(os.path.abspath(cfg.output_dir),"final"),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        
    return


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_full_model(cfg: DictConfig):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator()
    
    accelerator.wait_for_everyone()
    device= accelerator.device
   

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    model = load_model(cfg, cfg.model.from_trained)

    # Optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW #Adafactor
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)


    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
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
    if cfg.tokenizer.name is not None:
        train_loader = get_loader2(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json, cfg.dataset.train.file_list,
                tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.dataset.train_batch_size, s_max= cfg.dataset.tokens.s_max,
                n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn)
        
        eval_loader = get_loader2(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, cfg.dataset.eval.file_list,
                tokenizer_name = cfg.tokenizer.name, transform= transform, batch_size = cfg.dataset.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
                n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, use_tokenizer_fast=cfg.tokenizer.use_fast, collate_fn=collate_fn)
    else:
        vocabulary1 = load_vocab(cfg.vocabs.name1)
        vocabulary2 = load_vocab(cfg.vocabs.name2)
        train_loader = get_loader(cfg.dataset.train.image_dir, cfg.dataset.train.caption_json,cfg.dataset.train.history_json, cfg.dataset.train.file_list,
               vocabulary = vocabulary1, vocabulary2=vocabulary2, transform= transform, batch_size = cfg.dataset.train_batch_size, s_max= cfg.dataset.tokens.s_max,
               n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn)
    
        eval_loader = get_loader(cfg.dataset.eval.image_dir, cfg.dataset.eval.caption_json, cfg.dataset.eval.history_json, cfg.dataset.eval.file_list,
               vocabulary = vocabulary1, vocabulary2=vocabulary2, transform= transform, batch_size = cfg.dataset.eval_batch_size, s_max= cfg.dataset.tokens.s_max,
               n_max=cfg.dataset.tokens.n_max, shuffle=cfg.training.shuffle, collate_fn=collate_fn)


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
    custom_loss = CustomLoss()
    bce_loss = nn.BCELoss()

    for epoch in range(cfg.training.num_epochs):
        model.train()
                
        if cfg.tracking:
            total_loss = 0
        train_losses = []

        for step, (encoded_images,patient_history, labels, reports, true_stop_probs) in enumerate(train_loader):
            # encoded_images = encoded_images.to(device)
            # reports = reports.to(device)
            # true_stop_probs = true_stop_probs.to(device)

            loss = 0            
            n_sentences  = reports.shape[1]
            
            
            if model.co_attention:
                encoded_images , tags = model.encoder(encoded_images)
                semantic_features = model.semantic_features_extractor(tags)
            else:
                encoded_images = model.encoder(encoded_images)
            
            if model.history_encoder is not None:
                patient_history = model.history_encoder(patient_history)
                    
            prev_hidden, prev_cell_state = model.lstm.init_state(encoded_images)
            lstm_init = True

            #output = model(encoded_images, reports[:, :-1])  # [batch_size, seq_len - 1, vocab_size]
            for i in range(n_sentences):
                
                if model.co_attention:
                    if model.history_encoder is not None:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features, patient_history)
                    else:
                        context_vector, att_wts , _  = model.attention( prev_hidden, encoded_images, semantic_features)
                else:
                    # Attend to encoded_images
                    if model.history_encoder is not None:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images, patient_history)                          
                    else:
                        context_vector, att_wts = model.attention(prev_hidden, encoded_images)

                # Generate Topic Vector
                prev_hidden, pred_stop_probs = model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)  # [batch_size, d_model]
                lstm_init= False
                
                # Decode reports
                tgt = reports[:,i, :-1]  # Remove last token from reports
                padding_mask = create_padding_mask(tgt)
                #causal_mask1 = create_causal_masks(inputs)
                tgt_mask = src_mask(tgt.shape[1])
                
                memory = encoded_images * att_wts # [batch_size, seq_len, d_model]
                #memory_mask = None

                output = model.decoder(tgt, prev_hidden, memory,tgt_key_padding_mask=padding_mask, tgt_mask=tgt_mask,
                                       tgt_is_causal=True)  # [batch_size, seq_len - 1, d_model]
                
                loss += custom_loss(true_stop_probs, reports[:, i, 1:], pred_stop_probs,  output)  # Ignore <sos> token

            loss += bce_loss(tags, labels)
            
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
                eval_loss, eval_bce_loss, perplexity = evaluate_full(model, accelerator, eval_loader, custom_loss, bce_loss)
                logger.info(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} perplexity: {perplexity} eval_loss:{eval_loss} \
                    bce_loss {eval_bce_loss}")
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
                    output_dir = os.path.join(cfg.output_dir, epoch_dir)
                    
                    unwrapped_model.save_pretrained(
                        output_dir,
                        is_man_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model),
                    )

            if completed_steps >= cfg.training.max_train_steps:
                break
            
        eval_loss, eval_bce_loss, perplexity = evaluate_full(model,accelerator,eval_loader, custom_loss, bce_loss)
        model.train()
        logger.info(f"epoch {epoch}: train_loss: {train_loss} perplexity: {perplexity} eval_loss: {eval_loss} bce_loss: {eval_bce_loss}")
 
        
    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            os.path.join(os.path.abspath(cfg.output_dir),"final"),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        
    return



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_encoder(cfg: DictConfig):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator()
    
    accelerator.wait_for_everyone()
    device= accelerator.device
   

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    model = load_model(cfg, load_tokenizer=False)

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
        )


    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

  
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        cfg.training.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    
    # Create train_loader and eval_loader here
    train_loader = get_enc_loader(cfg.dataset.image_dir,
               cfg.dataset.file_list,
               cfg.dataset.transform,
               cfg.dataset.batch_size,
               shuffle=cfg.training.shuffle)

    eval_loader = get_enc_loader(cfg.dataset.image_dir,
               cfg.dataset.file_list,
               cfg.dataset.transform,
               cfg.dataset.batch_size,
               shuffle=cfg.training.shuffle)


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
    
    bce_loss = nn.BCELoss()

    for epoch in range(cfg.training.num_epochs):
        model.train()
        
        if cfg.tracking:
            total_loss = 0
            
        train_losses = []
        
        for step, (images, pathology) in enumerate(train_loader):
            loss = 0
            output = model(images)
            
            loss += bce_loss(output, pathology)
            
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
                eval_loss = evaluate_encoder(model, eval_loader, accelerator,  bce_loss)
                
                logger.info(f"Epoch {epoch}, Step {step} : train_loss: {train_loss} eval_loss:{eval_loss}")
                model.train()
            
            if step % cfg.training.save_every == 0:                 
                epoch_dir = f"epoch_{epoch}_most_recent"
                
                logger.info(f"Saving model in {epoch_dir}..")
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
                    
            if completed_steps >= cfg.training.max_train_steps:
                break
            
        eval_loss = evaluate_encoder(model, eval_loader, accelerator, bce_loss)
        model.train()
        logger.info(f"epoch {epoch}:  train_loss: {train_loss} eval_loss: {eval_loss}")
 
        
    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            os.path.join(os.path.abspath(cfg.output_dir),"final_encoder"),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )   
            
    return
            
            
def evaluate_encoder(model, eval_loader, accelerator, bce_loss):
    model.eval()
    eval_losses = []
    loss = 0
    for _, (images, pathology) in enumerate(eval_loader):
        output = model(images)
        
        loss = bce_loss(output, pathology)
        gathered_loss= accelerator.gather(loss).detach().cpu()
        eval_losses.append(gathered_loss)
        
    eval_loss =  torch.mean(torch.cat(eval_losses))
    
    return eval_loss



if __name__ == "__main__":
    
    #train_full_model()
    #train_encoder()
    train_decoder()
    
    
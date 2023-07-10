
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from pyciderevalcap.ciderD.ciderD import CiderD
from nltk.translate.meteor_score import meteor_score

class ImageCaptioningModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, vocab_size):
        super(ImageCaptioningModel, self).__init__()

        # Encoder: VGG16
        self.encoder = nn.Sequential(
            nn.Linear(512 * 14 * 14, d_model),
            nn.ReLU(),
        )

        # Attention Module: Bahdanau Attention
        self.attention = BahdanauAttention(d_model)

        # LSTM Layer
        self.lstm = LSTMLayer(d_model, d_model, num_layers)

        # Decoder: Transformer Decoder
        self.decoder = Decoder(d_model, nhead, dim_feedforward, num_layers)

        # Output Layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, images, captions):
        # images: [batch_size, 512, 14, 14]
        # captions: [batch_size, seq_len]

        # Encode Images
        images = images.reshape(images.size(0), -1)
        encoded_images = self.encoder(images)  # [batch_size, d_model]

        # Attend to Images
        context_vector, _ = self.attention(encoded_images.unsqueeze(1), captions)  # [batch_size, d_model]

        # Generate Topic Vector
        topic_vector = self.lstm(context_vector.unsqueeze(0))[0][-1]  # [batch_size, d_model]

        # Decode Captions
        tgt = captions[:, :-1]  # Remove last token from captions
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = encoded_images.unsqueeze(1).repeat(1, tgt.size(1), 1)  # [batch_size, seq_len, d_model]
        memory_mask = None

        output = self.decoder(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)  # [batch_size, seq_len - 1, d_model]
        output = self.output_layer(output)  # [batch_size, seq_len - 1, vocab_size]

        return output

def train(model, optimizer, scheduler, train_loader, valid_loader, accelerator):
    model.train()

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    rouge = Rouge()
    ciderd = CiderD(df='corpus')
    device = accelerator.device

    for epoch in range(epochs):
        train_loss = 0.0
        train_bleu = []
        train_rouge_l = []
        train_cider_n = []
        train_meteor = []

        for i, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            output = model(images, captions[:, :-1])  # [batch_size, seq_len - 1, vocab_size]

            loss = loss_fn(output.reshape(-1, output.size(-1)), captions[:, 1:].reshape(-1))  # Ignore <sos> token

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (i + 1) % gradient_accumulation_steps == 0:
                accelerator.update()

            if (i + 1) % eval_every == 0:
                model.eval()

                valid_loss = 0.0
                valid_bleu = []
                valid_rouge_l = []
                valid_cider_n = []
                valid_meteor = []

                with torch.no_grad():
                    for images, captions in valid_loader:
                        images = images.to(device)
                        captions = captions.to(device)

                        output = model(images, captions[:, :-1])  # [batch_size, seq_len - 1, vocab_size]

                        loss = loss_fn(output.reshape(-1, output.size(-1)), captions[:, 1:].reshape(-1))  # Ignore <sos> token

                        valid_loss += loss.item()

                        # Evaluate Metrics
                        predicted_captions = output.argmax(dim=-1).cpu().numpy().tolist()
                        target_captions = captions[:, 1:].cpu().numpy().tolist()

                        valid_bleu.append(corpus_bleu([[target_caption] for target_caption in target_captions], predicted_captions))
                        valid_rouge_l.append(sum([rouge.get_scores(predicted_caption, [target_caption])[0]['rouge-l']['f'] for predicted_caption, target_caption in zip(predicted_captions, target_captions)]) / len(predicted_captions))
                        valid_cider_n.append(ciderd.compute_score({i: [predicted_caption] for i, predicted_caption in enumerate(predicted_captions)}, {i: [target_caption] for i, target_caption in enumerate(target_captions)})[0])
                        valid_meteor.append(sum([meteor_score([predicted_caption], [target_caption]) for predicted_caption, target_caption in zip(predicted_captions, target_captions)]) / len(predicted_captions))

                train_loss /= eval_every * gradient_accumulation_steps
                valid_loss /= len(valid_loader)

                train_bleu.append(sum(valid_bleu) / len(valid_bleu))
                train_rouge_l.append(sum(valid_rouge_l) / len(valid_rouge_l))
                train_cider_n.append(sum(valid_cider_n) / len(valid_cider_n))
                train_meteor.append(sum(valid_meteor) / len(valid_meteor))

                print(f"Epoch {epoch + 1}, Step {i + 1}: Train Loss {train_loss:.4f}, Valid Loss {valid_loss:.4f}, Valid BLEU {train_bleu[-1]:.4f}, Valid ROUGE-L {train_rouge_l[-1]:.4f}, Valid CIDEr-N {train_cider_n[-1]:.4f}, Valid METEOR {train_meteor[-1]:.4f}")

                train_loss = 0.0

                model.train()











if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--load_in_4bit", type=bool, default=False)
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--load_in_16bit", type=bool, default=False)
    parser.add_argument("--device_map", type=str, default={"":0})
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)

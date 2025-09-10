import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import torch.utils.tensorboard
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from dataset import BilingualDataset, casual_mask
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import re

from tqdm import tqdm
import warnings

# Add after imports
def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu")


def get_all_sentences(ds, lang):
    """
    Extract all sentences from the dataset for a given language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenzier(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), 
            trainer = trainer
        )
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    tokenizer_src = get_or_build_tokenzier(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenzier(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max len of source sentence: {max_len_src}')
    print(f'Max len of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model']
    )
    return model

def _find_latest_checkpoint(model_folder: str, model_basename: str):
    folder = Path(model_folder)
    if not folder.exists():
        return None
    pattern = re.compile(rf"^{re.escape(model_basename)}(\d+)\.pt$")
    candidates = []
    for f in folder.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])

def train_model(config):
   
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f'Using device: {device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        if config["preload"] == "latest":
            model_filename = _find_latest_checkpoint(config['model_folder'], config['model_basename'])
        else:
            model_filename = get_weights_file_path(config, config['preload'])
        if model_filename and Path(model_filename).exists():
            print(f'Loading model from {model_filename}')
            state = torch.load(model_filename, map_location=device)
            initial_epoch = state.get('epoch', -1) + 1
            global_step = state.get('global_step', 0)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            print('No checkpoint found. Starting fresh.')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # run through transformers

            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)  
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)

            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # Compute CE loss with targets
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

            # log the loss
            writer.add_scalar('loss/train', loss.item(), global_step)
            writer.flush()
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    # Initialize GPU
    device = check_gpu()
    
    config = get_config()
    train_model(config)
    print("Training complete.")
    
    # Add cleanup for GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self, ds,tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # store ids (ints)
        self.sos_id = tokenizer_tgt.token_to_id('[SOS]')
        self.eos_id = tokenizer_tgt.token_to_id('[EOS]')
        self.pad_id = tokenizer_tgt.token_to_id('[PAD]')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too short for the input data.")

        encoder_input = torch.tensor(
            [self.sos_id] + enc_input_tokens + [self.eos_id] + [self.pad_id] * enc_num_padding_tokens,
            dtype=torch.int64
        )
        decoder_input = torch.tensor(
            [self.sos_id] + dec_input_tokens + [self.pad_id] * dec_num_padding_tokens,
            dtype=torch.int64
        )
        label = torch.tensor(
            dec_input_tokens + [self.eos_id] + [self.pad_id] * dec_num_padding_tokens,
            dtype=torch.int64
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,Seq_len)
        decoder_mask = (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0) & casual_mask(decoder_input.size(0))  # (1,Seq_len,Seq_len)

        return{
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def casual_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0
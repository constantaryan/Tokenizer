#!/usr/bin/env python
# coding: utf-8

import unicodedata
import regex as re
import tiktoken
import json
import os

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids[:], ids[1:]):   # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self):
        self.merges = {}  # (int,int) -> int
        self.pattern = ""  # string
        self.special_tokens = {}  # str -> int ,e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes
        
    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError 
    
    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
            
        return vocab

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self._load_pretrained_data()
    
    def _load_pretrained_data(self):
        """Load pre-trained merges and vocab from JSON file"""
        try:
            with open('basic_tokenizer.json', 'r') as f:
                data = json.load(f)
            
            # Convert string keys back to tuples for merges
            merges = {}
            for key, value in data['merges'].items():
                pair = tuple(map(int, key.split(',')))
                merges[pair] = value
            
            # Convert string keys back to integers for vocab
            vocab = {}
            for key, value in data['vocab'].items():
                vocab[int(key)] = value.encode('utf-8')
            
            self.merges = merges
            self.vocab = vocab
        except FileNotFoundError:
            # Fallback to empty if file not found
            self.merges = {}
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256 
        
        # input text pre-processing 
        text_bytes = text.encode("utf-8")  # raw bytes 
        ids = list(text_bytes)  # list of integers in range 0..255
        
        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            
            # mint a new token: assign it the next available id
            idx = 256 + i
            
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # print - easier for debugging 
            if verbose:
                print(f"merge {i+1}/{num_merges} : {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences")
        
        # save class variables for future use
        self.merges = merges  # it will be used in encode()
        self.vocab = vocab   # it will be used in decode()
        
    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
        
    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index -> it must be the latest merge 
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list,
            
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        
        return ids

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_token = {}
        self.inverse_special_token = {}
        self._load_pretrained_data()
    
    def _load_pretrained_data(self):
        """Load pre-trained merges and vocab from JSON file"""
        try:
            with open('regex_tokenizer.json', 'r') as f:
                data = json.load(f)
            
            # Convert string keys back to tuples for merges
            merges = {}
            for key, value in data['merges'].items():
                pair = tuple(map(int, key.split(',')))
                merges[pair] = value
            
            # Convert string keys back to integers for vocab
            vocab = {}
            for key, value in data['vocab'].items():
                vocab[int(key)] = value.encode('utf-8')
            
            # Load special tokens if they exist
            special_tokens = data.get('special_tokens', {})
            
            self.merges = merges
            self.vocab = vocab
            self.special_tokens = special_tokens
            self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        except FileNotFoundError:
            # Fallback to empty if file not found
            self.merges = {}
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            self.special_tokens = {}
            self.inverse_special_tokens = {}
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # split the text up into text chunks 
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunks_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunks_ids, stats)
            
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunks_ids, pair, idx) for chunks_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
            
        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()
            
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        
    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
                
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes):
        # first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids
    
    def encode_ordinary(self, text):
        # """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunks in text_chunks:
            chunks_bytes = chunks.encode("utf-8")
            chunks_ids = self._encode_chunk(chunks_bytes)
            ids.extend(chunks_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
            
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1::])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
                
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx+2:]
        
    return parts

def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, rank))
        assert len(pair) == 2
        rank1 = mergeable_ranks[pair[0]]
        rank2 = mergeable_ranks[pair[1]]
        merges[(rank1, rank2)] = rank
    return merges

GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__()
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)
        
    def _encode_chunk(self, text_bytes):
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

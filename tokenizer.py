import json
from typing import List, Dict
import regex as re

class CarolineTokenizer:
    def __init__(self, vocab_size: int = 102400):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self._build_vocab()
        
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    def _build_vocab(self):
        for i in range(self.vocab_size):
            token = f"<|byte_{i}|>"
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def encode(self, text: str) -> List[int]:
        tokens = []
        for match in self.compiled_pattern.finditer(text):
            token_str = match.group()
            if token_str in self.vocab:
                tokens.append(self.vocab[token_str])
            else:
                tokens.append(self.vocab["<|unk|>"])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.inverse_vocab.get(token, "") for token in tokens])

    def save_vocab(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    def load_vocab(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

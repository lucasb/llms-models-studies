class Tokenizer:
    def __init__(self) -> None:
        self.merges = {}
        self.spacials = {}
        self.vocab = {}

    def _build_vocab(self) -> dict:
        # base vocab with only chars
        vocab = {i: bytes([i]) for i in range(256)}
        # merge chars vocab to reduce context length needed
        for (idx1, idx2), i in self.merges.items():
            vocab[i] = vocab[idx1] + vocab[idx2]
        # sequence of chars to represent special behavor tokens
        for spacial_token, i in self.spacials.items():
            vocab[i] = spacial_token.encode("utf-8")
        return vocab

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return []

    def decode(self, ids: list[int]) -> str:
        return ""

    def save(self, model_name: str) -> None:
        pass

    def load(self, model_file: str) -> None:
        pass

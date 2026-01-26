class Tokenizer:
    def __init__(self) -> None:
        pass

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

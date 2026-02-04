# get pair from a list of integer
# ex. [1,2,3,1,2] -> {(1,2): 2. (2,3): 1, (3,1): 1}
def get_pair_frequency_counts(
    vocab_ids: list[int], counts_to_update: dict = {}
) -> dict:
    counts = counts_to_update
    vocab_ids_length = len(vocab_ids)
    for i in range(vocab_ids_length):
        if i < vocab_ids_length - 1:
            # get current and next id to build pair
            pair = (vocab_ids[i], vocab_ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


# replace all pair occurency with new single id
# ex. [1,2,3,1,2], (1,2), 4 -> [4,3,4]
def merge_pair(vocab_ids: list[int], pair: tuple[int, int], pair_id: int) -> list[int]:
    merged_ids = []
    i = 0
    while i < len(vocab_ids):
        if vocab_ids[i] == pair[0] and vocab_ids[i + 1] == pair[1]:
            merged_ids.append(pair_id)
            i = i + 2
        else:
            merged_ids.append(vocab_ids[i])
            i = i + 1
    return merged_ids


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


if __name__ == "__main__":
    file = "wikipedia_tartan.txt"
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        text_bytes = content.encode("utf-8")  # convert to bytes
        print(max(text_bytes))

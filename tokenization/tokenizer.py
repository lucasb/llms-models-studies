import regex as re

# partern to split tokens
GPT4LIKE_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# get pair from a list of integer
# ex. [1,2,3,1,2] -> {(1,2): 2. (2,3): 1, (3,1): 1}
def get_pair_frequency_counts(
    vocab_ids: list[int], counts_to_update: dict = None
) -> dict:
    counts = {} if counts_to_update is None else counts_to_update
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
        if (
            i < len(vocab_ids) - 1
            and vocab_ids[i] == pair[0]
            and vocab_ids[i + 1] == pair[1]
        ):
            merged_ids.append(pair_id)
            i = i + 2
        else:
            merged_ids.append(vocab_ids[i])
            i = i + 1
    return merged_ids


class Tokenizer:
    def __init__(self) -> None:
        self.pattern = re.compile(GPT4LIKE_SPLIT_PATTERN)
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
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into chunks
        text_chunks = re.findall(self.pattern, text)

        # vocab ids from training text
        text_vocab_ids = [list(txt_ch.encode("utf-8")) for txt_ch in text_chunks]

        vocab = self._build_vocab()
        merges = {}  # pairs(int, int), pair_id(int)

        for i in range(num_merges):
            # count the pair frequency
            pair_frequency = {}
            for txt_chunks in text_vocab_ids:
                get_pair_frequency_counts(txt_chunks, pair_frequency)

            # more frequenty pair
            # key need to look to value max insted of key max
            pair = max(pair_frequency, key=pair_frequency.get)
            vocab_pair_id = len(vocab)

            # add on merge list and vocab
            merges[pair] = vocab_pair_id
            # save the concatened pair bytes in the new index
            vocab[vocab_pair_id] = vocab[pair[0]] + vocab[pair[1]]

            # update text_vocab_ids from training text
            text_vocab_ids = [
                merge_pair(txt_chunks, pair, vocab_pair_id)
                for txt_chunks in text_vocab_ids
            ]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {vocab_pair_id} ({vocab[vocab_pair_id]}) had {pair_frequency[pair]} occurences"
                )

        self.merges = merges  # to use in encode()
        self.vocab = vocab  # to us in decode()

    def encode(self, text: str) -> list[int]:
        # encode the chunck
        def _encode_chunks(text_bytes):
            vocab_ids = list(text_bytes)
            if len(vocab_ids) > 1:
                # local copy to iterate
                merges_sorted = dict(
                    sorted(self.merges.items(), key=lambda item: item[1])
                )
                for pair, pair_id in merges_sorted.items():
                    vocab_ids = merge_pair(vocab_ids, pair, pair_id)
            return vocab_ids

        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = _encode_chunks(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def decode(self, vocab_ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[vocab_id] for vocab_id in vocab_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, model_name: str) -> None:
        pass

    def load(self, model_file: str) -> None:
        pass


if __name__ == "__main__":
    text = "aaaabbaaacccc"
    tokenizer = Tokenizer()

    file = "wikipedia_tartan.txt"
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        tokenizer.train(content, 5000, True)

    print("----")
    file = "wikipedia_python.txt"
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        text = content[:1000]
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc)
        print(text)
        print(enc)
        print(dec)
        print(len(enc))

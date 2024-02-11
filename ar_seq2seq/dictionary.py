from fairseq.data import Dictionary
from fairseq.file_io import PathManager


class T5Dictionary(Dictionary):
    def __init__(
            self,
            *,  # begin keyword-only arguments
            pad="<pad>",
            eos="</s>",
            unk="<unk>",
            extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = pad, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        # self.bos_index = self.add_symbol(pad)
        # self.pad_index = self.add_symbol(pad)
        # self.eos_index = self.add_symbol(eos)
        # self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols) + 3

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols,
                ex_vals + self.count,
            ),
        )

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                word_idx = self.add_symbol(word, n=count, overwrite=overwrite)
                if word == self.bos_word:
                    self.bos_index = word_idx
                if word == self.eos_word:
                    self.eos_index = word_idx
                if word == self.pad_word:
                    self.pad_index = word_idx
                if word == self.unk_word:
                    self.unk_index = word_idx
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

class TgtDictionary4T5Encoder(Dictionary):
    def __init__(
            self,
            *,  # begin keyword-only arguments
            bos="<pad>",
            pad="<pad>",
            eos="</s>",
            unk="<unk>",
            extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

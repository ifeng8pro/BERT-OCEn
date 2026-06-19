from pytorch_pretrained_bert import BertTokenizer
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX

import torch
# BertTokenizer reserved tokens: "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"


class MyBertTokenizer(BertTokenizer):
    """ Patch of pytorch_pretrained_bert.BertTokenizer to fit torchnlp TextEncoder() interface. """

    def __init__(self, vocab_file, do_lower_case=True, append_eos=False, add_special_tokens=True):
        super().__init__(vocab_file, do_lower_case=do_lower_case)
        self.append_eos = append_eos
        self.add_special_tokens = add_special_tokens  # Add new parameter to control special token addition

        self.itos = list(self.vocab.keys())
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.vocab = self.itos
        self.vocab_size = len(self.vocab)

    def encode(self, text, eos_index=DEFAULT_EOS_INDEX, unknown_index=DEFAULT_UNKNOWN_INDEX):
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        text = self.tokenize(text)
        unknown_index = self.stoi['[UNK]']  # overwrite unknown_index according to BertTokenizer vocab

        # Add BERT special tokens
        if self.add_special_tokens:
            # [CLS] + tokens + [SEP]
            vector = [self.stoi['[CLS]']]  # CLS token at the beginning
            vector.extend([self.stoi.get(token, unknown_index) for token in text])
            vector.append(self.stoi['[SEP]'])  # SEP token at the end
        else:
            # Original logic (no special tokens added)
            vector = [self.stoi.get(token, unknown_index) for token in text]

        if self.append_eos:
            vector.append(eos_index)

        return torch.LongTensor(vector)

    def decode(self, tensor):
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text. """
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)

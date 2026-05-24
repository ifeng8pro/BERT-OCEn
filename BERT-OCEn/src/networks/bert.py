import torch.nn as nn
import torch
from pytorch_pretrained_bert.modeling import BertModel


class BERT(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self,pretrained_model_name='bert-base-uncased', cache_dir='../data/bert_cache/english',finetune=0):
        super().__init__()
        # Check if choice of pretrained model is valid

        assert pretrained_model_name in ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased','bert-base-chinese')
        # Load pre-trained BERT model
        self.bert_base = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        if finetune > 0:

            save_path = "../data/bert_cache/chinese/finetuned/finetuned_bert_model.pth"
            # Load fine-tuned weights
            finetuned_state_dict = torch.load(save_path)
            # Update parameters of the original BERT model
            self.bert_base.load_state_dict(finetuned_state_dict)
            print("Finetuned BERT model loaded.")
        else:
            print("Finetuned BERT model not loaded.")


        self.bert = self.bert_base
        self.embedding = self.bert_base.embeddings
        self.embedding_size = self.embedding.word_embeddings.embedding_dim

        # Remove BERT model parameters from optimization
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, x):
        # Transpose input to match BERT format
        input_ids = x.transpose(0, 1)  # (batch_size, sentence_length)

        # Create attention mask (important!)
        attention_mask = (input_ids != 0).long()  # 0 represents padding

        # Pass to BERT
        hidden, _ = self.bert(input_ids,
                              attention_mask=attention_mask,
                              output_all_encoded_layers=False)
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        # Check if the first token is CLS (101)
        first_tokens = input_ids[:, 0]  # First token of each sample
        is_cls = torch.all(first_tokens == 101)   # Check if all are CLS tokens

        if is_cls:
            # Use CLS vector: first token of each sample
            sentence_vectors = hidden[:, 0, :]  # (batch_size, hidden_size)
        else:
            print('CLS token not found')

        return sentence_vectors

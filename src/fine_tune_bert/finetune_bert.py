import torch.nn as nn
from pytorch_pretrained_bert import BertModel,BertTokenizer
import pandas as pd
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import time

# # Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define absolute path for BERT
# Replace with the path to your local bert-base-chinese model directory.
bert_dir = r"D:\Student\202507sci\github\CVDD-PyTorch-master\data\bert_cache\chinese"  
bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-chinese', cache_dir=bert_dir)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese', cache_dir=bert_dir)

# Record initial parameters before training
initial_params = {name: param.clone().detach() for name, param in bert.named_parameters() if param.requires_grad}


# 2. Define BERT fine-tuning model with binary classification head
class FineTunedBERT(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_labels=2, dropout_prob=0.1):
        super(FineTunedBERT, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        # Get BERT output
        encoded_layers, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )

        # Add dropout
        pooled_output = self.dropout(pooled_output)

        # Classification logits
        logits = self.classifier(pooled_output)

        return encoded_layers, logits

# 3. Create fine-tuning model instance
model = FineTunedBERT(bert_model=bert)
model = model.to(device)  # Move entire model to GPU


# 4. Actual training process
def real_training(model, train_texts, train_labels, tokenizer, batch_size=32, max_len=128, epochs=3):
    # Check GPU availability
    model = model.to(device) # Move entire model to GPU


    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0,
                       float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    # Convert data to BERT input format
    def encode_texts(texts, labels):
        input_ids = []
        attention_masks = []
        label_tensors = []

        for text, label in zip(texts, labels):
            # Use legacy tokenizer encoding method
            tokenized_text = tokenizer.tokenize(text)
            if len(tokenized_text) > max_len - 2:  # Considering [CLS] and [SEP]
                tokenized_text = tokenized_text[:max_len - 2]

            # Add special tokens and convert to ids
            tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)

            # Pad/truncate to fixed length
            padding_length = max_len - len(input_id)
            attention_mask = [1] * len(input_id) + [0] * padding_length
            input_id = input_id + [0] * padding_length

            input_ids.append(torch.tensor(input_id))
            attention_masks.append(torch.tensor(attention_mask))
            label_tensors.append(torch.tensor(label))

        # Stack all samples
        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)
        labels = torch.stack(label_tensors).to(device)

        return input_ids, attention_masks, labels

    # Encode training data
    input_ids, attention_masks, labels = encode_texts(train_texts, train_labels)



    # Create DataLoader
    train_data = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,  # Keep shuffling
        num_workers=0,  # Set to 0 if data is already on GPU
        pin_memory=False,  # Important: data is already on GPU, set to False!
        drop_last=False
    )

    """    Used to control which BERT layers to train    """
    # 1. First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Only unfreeze parameters of the 12th layer (last layer)
    # BERT encoder layers are numbered 0-11 (total 12 layers), layer 12 is encoder.layer.11
    for name, param in model.named_parameters():
        if "encoder.layer.0" in name:
            param.requires_grad = True
        if "encoder.layer.1" in name:
            param.requires_grad = True
        if "encoder.layer.2" in name:
            param.requires_grad = True
        if "encoder.layer.3" in name:
            param.requires_grad = True
        if "encoder.layer.4" in name:
            param.requires_grad = True
        if "encoder.layer.5" in name:
            param.requires_grad = True
        if "encoder.layer.6" in name:
            param.requires_grad = True
        if "encoder.layer.7" in name:
            param.requires_grad = True
        if "encoder.layer.8" in name:
            param.requires_grad = True
        if "encoder.layer.9" in name:
            param.requires_grad = True
        if "encoder.layer.10" in name:
            param.requires_grad = True
        if "encoder.layer.11" in name:
            param.requires_grad = True
    # 2. Parameter grouping (with weight decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    # 3. Optimizer and scheduler
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    #scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # Prepare optimizer and loss function
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    # Add warmup step
    # Record warmup start time
    warmup_start = time.time()
    # Training loop
    print(f"begin training，there are {epochs} epoch...")
    print("!The first batch loading is time-consuming. waiting!")
    # Start loop training model (main fine-tuning process)
    for epoch in range(epochs): #epochs=4
        model.train()
        total_loss = 0
        for batch in train_loader: #batch=32
            batch_input_ids, batch_attention_mask, batch_labels = batch
            optimizer.zero_grad() # Clear previous gradients (avoid accumulation)
            # Forward pass
            _, logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            # Calculate loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            # Backward propagation
            loss.backward()  # Calculate gradients: ∂loss/∂w, result stored in .grad
            optimizer.step()  # Update parameters: w = w - lr * ∂loss/∂w
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")


def prepare_data_from_train_set(train_set, normal_classes):
    """
    Prepare training data from train_set.
    Labels in normal_classes are set to 0, other labels are set to 1.
    """
    train_texts = []
    train_labels = []

    # Count class distribution
    class_distribution = {}

    for example in train_set:
        text = str(example['text'])  # Ensure it's a string
        label = str(example['label'])  # Original class label

        # Record class distribution
        if label in class_distribution:
            class_distribution[label] += 1
        else:
            class_distribution[label] = 1

        # Set binary labels based on normal_classes
        if label in normal_classes:
            # Normal class, set label to 0
            train_labels.append(0)
        else:
            # Abnormal class, set label to 1
            train_labels.append(1)

        train_texts.append(text)

    return train_texts, train_labels


# 5. Save fine-tuned BERT weights (only BERT part)
def save_finetuned_bert(model, save_path):
    # Extract BERT part's state_dict
    bert_state_dict = {k.replace('bert.', ''): v for k, v in model.state_dict().items() if k.startswith('bert.')}
    # Save as new BERT model
    torch.save(bert_state_dict, save_path)
    return save_path



# 7. Main function: Fine-tune BERT on train_set and test_set
def fine_tune_from_datasets(train_set, normal_classes, epochs=10, batch_size=32, max_len=128):

    """
    Fine-tune BERT from train_set.
    Args:
        train_set: Training dataset
        normal_classes: List of normal classes, these are set to label 0, others to 1

    """

    print(f"\n=== Start BERT fine-tuning for classes: {normal_classes} ===")

    save_path = "../data/bert_cache/chinese/finetuned/finetuned_bert_model.pth"

    # Prepare training data
    train_texts, train_labels = prepare_data_from_train_set(train_set, normal_classes)
    label_counts = Counter(train_labels)
    print(
        f"Fine-tuning BERT training data distribution: 0(normal): {label_counts.get(0, 0)}, 1(abnormal): {label_counts.get(1, 0)}")
    # BERT training process
    real_training(
        model=model,
        train_texts=train_texts,
        train_labels=train_labels,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        epochs=epochs
    )
    # Save fine-tuned BERT weights
    save_path = save_finetuned_bert(model, save_path)

    print("=== BERT fine-tuning completed, new model has been saved ===\n")
    return save_path



import torch
import argparse
import os

from .train_utils import train_model, evaluate_model
from .model import JudgeModelV2
from .dataloader import make_dataloader, load_dataset
from .config import TrainConfig, load_config, save_config


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_dataset(dataset_paths):
    total_dataset = {}
    for path in dataset_paths:
        dataset = load_dataset(path)
        total_dataset.update(dataset)
    return total_dataset


def make_dataloaders(config: TrainConfig, tokenizer):
    assert config.train_dataset is not None, "train_dataset must be specified"
    assert config.val_dataset is not None, "val_dataset must be specified"
    assert config.test_dataset is not None, "test_dataset must be specified"
    train_set = _load_dataset(config.train_dataset)
    val_set = _load_dataset(config.val_dataset)
    test_set = _load_dataset(config.test_dataset)

    train_dataloader = make_dataloader(train_set, tokenizer, config.micro_batch_size)
    val_dataloader = make_dataloader(val_set, tokenizer, config.micro_batch_size)
    test_dataloader = make_dataloader(test_set, tokenizer, config.micro_batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def main(config: TrainConfig):
    model = JudgeModelV2(n_classes=3, dropout_prob=config.dropout, num_decoder_layers=config.n_blocks)

    if config.all_params:
        for p in model.parameters():
            p.requires_grad = True

    n_params = count_trainable_params(model) / 1e6
    print(f"Number of trainable parameters (millions): {n_params:.4f}")

    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(config, model.tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = config.device

    # Move model to the appropriate device
    model.to(device)

    init_f1 = 0.0
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint, map_location=device))
        accuracy, precision, recall, init_f1 = evaluate_model(model, val_dataloader, device)
        print(f"Initial Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {init_f1:.4f}\n")

    best_model = train_model(
        model, 
        train_dataloader, val_dataloader, 
        optimizer, loss_fn,
        config=config,
        init_f1=init_f1
    )

    accuracy, precision, recall, f1 = evaluate_model(best_model, test_dataloader, device)
    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    config_dict, config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config_dict)

    main(config)
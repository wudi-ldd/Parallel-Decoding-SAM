import os
import torch

# Configuration dictionary
CONFIG = {
    'dataset_name': 'pore',  # Dataset name, only this needs to be set
    'data_base_dir': 'datasets',  # Base directory for datasets
    'log_dir_base': 'logs',       # Base directory for logs
    'log_file': 'best_model_metrics.log',  # Log file name
    'save_dir_base': 'logs',      # Base directory for saving models
    'save_prefix': 'best_model',  # Prefix for saved model files
    'model_type': 'vit_l',        # Model type (e.g., ViT Large)
    'checkpoint': 'weights/sam_vit_l_0b3195.pth',  # Path to pretrained SAM checkpoint
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device (GPU or CPU)
    'num_epochs': 100,            # Number of training epochs
    'learning_rate': 1e-4,        # Learning rate
    'betas': (0.9, 0.999),        # Betas for AdamW optimizer
    'weight_decay': 1e-4,         # Weight decay for regularization
    'metric_weights': {           # Weights for composite score metrics
        'iou': 0.25,
        'f1': 0.25,
        'precision': 0.25,
        'recall': 0.25
    },
    'num_classes': 1,             # Number of output classes (binary segmentation)
    'batch_size': 1,              # Batch size

    # ----- Additional LoRA hyper-parameters -----
    'rank': 1,                    # LoRA rank
    'lora_alpha': 1,              # LoRA scaling factor; match rank to keep the delta stable.
    'lora_dropout': 0.0,          # Dropout probability on LoRA paths; set 0.0 to disable.
}

# Construct paths based on dataset_name
image_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'images')
mask_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'masks')
train_txt = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'train.txt')
val_txt = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'test.txt')

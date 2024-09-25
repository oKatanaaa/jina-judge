from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class JudgeModel(nn.Module):
    def __init__(self, n_classes, hidden_dim=512, dropout_prob=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder.lora_main_params_trainable = True
        # MLP for classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),  # First layer
            nn.LayerNorm(hidden_dim),  # Normalization layer
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout_prob),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),  # Second layer
            nn.LayerNorm(hidden_dim // 2),  # Normalization layer
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout_prob),  # Dropout for regularization
            nn.Linear(hidden_dim // 2, n_classes)  # Final layer mapping to number of classes
        )
    
    def to(self, device):
        self.device = device
        self.classification_head = self.classification_head.to(device)
        self.encoder = self.encoder.to(device)
        return self

    def forward(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        out =  self.encoder(**inputs.to(self.device))
        embedings = mean_pooling(out, inputs["attention_mask"])
        logits = self.classification_head(embedings)
        return logits
    

class JudgeModelV2(nn.Module):
    def __init__(self, n_classes, hidden_dim=512, num_decoder_layers=4, nhead=8, dropout_prob=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder.lora_main_params_trainable = True

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.encoder.config.hidden_size,
            nhead=nhead,
            dim_feedforward=self.encoder.config.hidden_size,
            dropout=dropout_prob
        )
        
        # Transformer Decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Embedding for a single token as the initial input to the decoder
        self.decoder_input_embedding = nn.Parameter(
            torch.randn(1, 1, self.encoder.config.hidden_size)
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, n_classes)
        )
    
    def to(self, device):
        self.device = device
        self.classification_head = self.classification_head.to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.decoder_input_embedding = nn.Parameter(self.decoder_input_embedding.to(device))
        return self
    
    def create_padding_mask(self, attention_mask):
        # Create mask: (batch_size, 1, seq_len) for use in decoder
        return attention_mask == 0

    def forward(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)

        encoder_outputs = self.encoder(**inputs.to(self.device))
        encoder_hidden_states = encoder_outputs.last_hidden_state.float()  # Shape: (batch_size, seq_len, hidden_dim)

        # Create attention masks for encoder and decoder
        encoder_padding_mask = self.create_padding_mask(inputs["attention_mask"]).to(self.device)
        
        # Create a batch of initial decoder inputs
        batch_size = encoder_hidden_states.size(0)
        decoder_input = self.decoder_input_embedding.expand(1, batch_size, -1).to(self.device)

        # Pass through decoder with cross-attention to encoder embeddings
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_hidden_states.transpose(0, 1),  # Transpose to (seq_len, batch_size, hidden_dim)
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=encoder_padding_mask
        ).squeeze(0)  # Remove the sequence dimension
        
        # Pass decoder output to the classification head
        logits = self.classification_head(decoder_output)
        return logits

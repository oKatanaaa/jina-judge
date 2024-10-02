from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
    

class JudgeModelV2(nn.Module):
    def __init__(self, n_classes, hidden_dim=512, num_decoder_layers=5, nhead=8, dropout_prob=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.encoder.lora_main_params_trainable = True

        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_prob
        )
        
        # Transformer Decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Embedding for a single token as the initial input to the decoder
        self.decoder_input_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim,)
        )

        # Classification head
        self.classification_head = nn.Linear(hidden_dim, n_classes)

    @property
    def device(self):
        return self.decoder_input_embedding.device
    
    def create_padding_mask(self, attention_mask):
        # Create mask: (batch_size, 1, seq_len) for use in decoder
        return attention_mask == 0

    def forward(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)

        encoder_outputs = self.encoder(**inputs)
        encoder_hidden_states = encoder_outputs.last_hidden_state.float()  # Shape: (batch_size, seq_len, hidden_dim)
        encoder_hidden_states = self.projection(encoder_hidden_states)

        # Create attention masks for encoder and decoder
        encoder_padding_mask = self.create_padding_mask(inputs["attention_mask"])
        
        # Create a batch of initial decoder inputs
        batch_size = encoder_hidden_states.size(0)
        decoder_input = self.decoder_input_embedding.expand(1, batch_size, -1)

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

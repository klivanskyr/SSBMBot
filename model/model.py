import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MarthTransformer(nn.Module):
    def __init__(
        self,
        state_dim=48,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        seq_len=16
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input embedding
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action prediction heads
        self.joystick_head = nn.Linear(d_model, 17)  # 17 discrete joystick positions
        self.cstick_head = nn.Linear(d_model, 9)     # 9 discrete c-stick positions
        self.trigger_l_head = nn.Linear(d_model, 3)  # 3 trigger states (0%, 50%, 100%)
        self.trigger_r_head = nn.Linear(d_model, 3)  # 3 trigger states (0%, 50%, 100%)
        self.buttons_head = nn.Linear(d_model, 8)    # 8 binary button outputs
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, state_seq):
        """
        Args:
            state_seq: [batch_size, seq_len, state_dim]
        
        Returns:
            Dictionary with action predictions:
            - joystick: [batch, 17] logits
            - cstick: [batch, 9] logits
            - trigger_l: [batch, 3] logits (0%, 50%, 100%)
            - trigger_r: [batch, 3] logits (0%, 50%, 100%)
            - buttons: [batch, 8] logits (for BCE loss)
        """
        # Project input to d_model dimension
        x = self.input_projection(state_seq)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # Use last token output (could also try mean pooling)
        x = x[:, -1, :]  # [batch, d_model]
        
        # Predict actions with each head
        outputs = {
            'joystick': self.joystick_head(x),      # [batch, 17]
            'cstick': self.cstick_head(x),          # [batch, 9]
            'trigger_l': self.trigger_l_head(x),    # [batch, 3]
            'trigger_r': self.trigger_r_head(x),    # [batch, 3]
            'buttons': self.buttons_head(x)         # [batch, 8]
        }
        
        return outputs
    
    def predict(self, state_seq):
        """
        Get discrete action predictions (for inference).
        
        Args:
            state_seq: [batch_size, seq_len, state_dim]
        
        Returns:
            Dictionary with discrete action indices/values
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(state_seq)
            
            predictions = {
                'joystick': torch.argmax(outputs['joystick'], dim=1),
                'cstick': torch.argmax(outputs['cstick'], dim=1),
                'trigger_l': torch.argmax(outputs['trigger_l'], dim=1),
                'trigger_r': torch.argmax(outputs['trigger_r'], dim=1),
                'buttons': torch.sigmoid(outputs['buttons']) > 0.5
            }
            
        return predictions
    
    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiHeadLoss(nn.Module):
    """Combined loss for all action heads."""
    
    def __init__(self, weights=None):
        super().__init__()
        
        # Default weights for each head
        if weights is None:
            weights = {
                'joystick': 1.0,
                'cstick': 1.0,
                'trigger_l': 0.3,
                'trigger_r': 0.3,
                'buttons': 0.5
            }
        self.weights = weights
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with model outputs (logits)
            targets: Dict with ground truth labels
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}
        
        # Joystick loss (17-way classification)
        loss_dict['joystick'] = self.ce_loss(
            predictions['joystick'], 
            targets['joystick'].squeeze()
        )
        
        # C-stick loss (9-way classification)
        loss_dict['cstick'] = self.ce_loss(
            predictions['cstick'],
            targets['cstick'].squeeze()
        )
        
        # Trigger L loss (3-way classification)
        loss_dict['trigger_l'] = self.ce_loss(
            predictions['trigger_l'],
            targets['trigger_l'].squeeze()
        )
        
        # Trigger R loss (3-way classification)
        loss_dict['trigger_r'] = self.ce_loss(
            predictions['trigger_r'],
            targets['trigger_r'].squeeze()
        )
        
        # Button loss (multi-label binary classification)
        loss_dict['buttons'] = self.bce_loss(
            predictions['buttons'],
            targets['buttons']
        )
        
        # Weighted total loss
        total_loss = sum(loss_dict[k] * self.weights[k] for k in loss_dict.keys())
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


# Helper function to create model with default hyperparameters
def create_marth_transformer(
    state_dim=48,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    seq_len=16
):
    """
    Factory function to create a MarthTransformer model.
    
    Args:
        state_dim: Input feature dimension (default: 48)
        d_model: Transformer hidden dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dim_feedforward: FFN hidden dimension (default: 1024)
        dropout: Dropout probability (default: 0.1)
        seq_len: Sequence length (default: 16)
    
    Returns:
        MarthTransformer model instance
    """
    model = MarthTransformer(
        state_dim=state_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        seq_len=seq_len
    )
    
    print(f"Created MarthTransformer with {model.get_num_params():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MarthTransformer...")
    
    # Create model
    model = create_marth_transformer(
        state_dim=48,
        d_model=256,
        nhead=8,
        num_layers=4,
        seq_len=16
    )
    
    # Create dummy input
    batch_size = 32
    seq_len = 16
    state_dim = 48
    
    dummy_input = torch.randn(batch_size, seq_len, state_dim)
    
    # Forward pass
    print("\nTesting forward pass...")
    outputs = model(dummy_input)
    
    print("\nOutput shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test predict method
    print("\nTesting predict method...")
    predictions = model.predict(dummy_input)
    
    print("\nPrediction shapes:")
    for key, val in predictions.items():
        print(f"  {key}: {val.shape}")
    
    # Test loss
    print("\nTesting loss computation...")
    criterion = MultiHeadLoss()
    
    # Create dummy targets - UPDATED for 3 trigger bins
    targets = {
        'joystick': torch.randint(0, 17, (batch_size, 1)),
        'cstick': torch.randint(0, 9, (batch_size, 1)),
        'trigger_l': torch.randint(0, 3, (batch_size, 1)),  # Changed to 3
        'trigger_r': torch.randint(0, 3, (batch_size, 1)),  # Changed to 3
        'buttons': torch.randint(0, 2, (batch_size, 8)).float()
    }
    
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"\nTotal loss: {loss.item():.4f}")
    print("Individual losses:")
    for key, val in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {val.item():.4f}")
    
    print("\n✓ Model tests passed!")
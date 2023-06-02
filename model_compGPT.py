# https://github.com/karpathy/minGPT/tree/master/mingpt
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.optim as optim
import numpy as np


batch_size = 64
seq_len = 100
num_tokens = 10000
d_model = 512
num_layers = 6
num_heads = 4
d_ff = 2048
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):    
    """Positional Encoding for the Transformer model."""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        # angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        angle_rates = torch.pow(10000, (2 * (torch.div(indexes,2,rounding_mode='floor'))) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        # position_encoding = torch.zeros(max_seq_len, hidden_dim)
        # position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        positions = torch.arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device)
        indexes = torch.arange(input_sequences.size(2)).unsqueeze(0).to(input_sequences.device)
        angles = self.get_angles(positions, indexes)

        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1)
        return position_encoding[:, :seq_len, :]

def FeedForward(d_model, d_ff, dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(), #Todo Use GELU instead?
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout)
    )
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Multi-Head Attention
        self.multi_head_attention = MultiheadAttention(d_model, num_heads, dropout) 
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        # Feedforward Layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Multi-Head Attention
        # mask = mask.transpose(0, 1) # [batch_size, seq_len, seq_len]
        attention_output = self.multi_head_attention(x, x, x)
        x = x + attention_output[0] # residual connection
        x = self.layer_norm1(x)
        
        # Feedforward Layer
        feedforward_output = self.feed_forward(x)
        x = x + feedforward_output
        x = self.layer_norm2(x)
        
        return x
class GPT3Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(GPT3Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding Layer
        # self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) 
        
        # Final Linear Layernu
        # self.fc = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model * 12, d_model)
        self.fc3 = nn.Linear(d_model,1)
        self.activation = nn.Tanh()
    def forward(self, x, mask=None):
        # Embedding Layer
        # x = self.token_embedding(x)
        # print("start", x.size(),x)
        pos = self.positional_encoding(x)
        # print("self.positional_encoding(x)",pos, pos.size())
        x = x + pos
        # print("x + pos",x.size(),x)
        # Decoder Layers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, mask) 
        
        # Final Linear Layer
        # print("self.decoder_layers[i](x, mask)",x.size(),x)
        x = self.fc(x)
        # print("self.fc(x)", x.size(),x)
        x = torch.flatten(x, 1)
        # print("torch.flatten(x, 1)",x,x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        # print("self.fc2(x)",x,x.size())
        x = self.activation(x)
        return x
    
def train(data):
    # Define hyperparameters
    epochs = 10
    lr = 0.001

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Instantiate model and optimizer
    model = GPT3Decoder(num_tokens, d_model, num_layers, num_heads, d_ff, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Start training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(data)):
            # Get inputs and targets
            inputs = data[i][0].to(device)
            targets = data[i][1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.view(-1, num_tokens), targets.view(-1))

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
if __name__ == '__main__':    


    # create a predictable sequence with a linear trend
    x = np.linspace(0, 200, seq_len)
    # x_batch = np.tile(x, batch_size).reshape(batch_size, seq_len)
    # x = torch.from_numpy(x_batch).int()
    # # x = torch.randint(0, num_tokens, (batch_size, seq_len))
    # print(x)
    
    # mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    # mask = torch.triu(mask, diagonal=1)
    # model = GPT3Decoder(num_tokens, d_model, num_layers, num_heads, d_ff, dropout)
    # out = model(x, mask)
    # # print(x[0])
    # print(out[0])
    # print(out.shape)

    # train(x)


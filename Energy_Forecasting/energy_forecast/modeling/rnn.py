import torch
from torch import nn

class RNNfromscratch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout

        self.W_xh = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(self.hidden_size))

        if self.num_layers > 1:
            self.W_layers = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers-1)])
            self.U_layers = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers-1)])
            self.b_layers = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(self.num_layers-1)])

        self.dropout = nn.Dropout(self.dropout_prob)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Function called by Pytorch framework to init parameters.
        """
        nn.init.xavier_normal_(self.W_xh) # For input-to-hidden with corrected variance
        nn.init.orthogonal_(self.W_hh) # for recurrent weights
        nn.init.zeros_(self.b_h)
        
        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                nn.init.xavier_uniform_(self.W_layers[i])
                nn.init.orthogonal_(self.U_layers[i])
                nn.init.zeros_(self.b_layers[i])

    
    def forward(self, X, state=None):
        # X.shape = (seq_len, batch_size, input size)
        seq_len, batch_size, _ = X.shape

        if state is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=X.device)
        else:
            # Clone to avoid inplace operations which can cause errors during backprop
            hidden = state.clone()

        outputs = torch.zeros(seq_len, batch_size, self.hidden_size, device=X.device)

        for t in range(seq_len):
            new_hidden = torch.zeros_like(hidden)
            x_t = X[t]
            for layer in range(self.num_layers):
                if layer==0:
                    h_state = torch.tanh(x_t @ self.W_xh + hidden[layer] @ self.W_hh + self.b_h)
                else:
                    h_state = torch.tanh(hidden[layer] @ self.U_layers[layer-1] + hidden[layer-1] @ self.W_layers[layer-1] + self.b_layers[layer-1])

                if layer < self.num_layers-1 or t < seq_len-1:
                    h_state = self.dropout(h_state)

                new_hidden[layer] = h_state
            # print(f"For time step {t} hidden state shape: {len(hidden)}")
            hidden = new_hidden.clone()
            outputs[t] = hidden[-1]

        return outputs, new_hidden


class RNNForecasting(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, forecast_steps):
        super().__init__()
        self.output_size = output_size
        self.forecast_steps = forecast_steps
        self.rnn = RNNfromscratch(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.output_layers = nn.ModuleList(nn.Linear(hidden_size, output_size) for _ in range(forecast_steps))
        # Normalisation de couche pour stabiliser l'apprentissage
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, X):
        X = X.permute(1, 0, 2)
        _, hidden = self.rnn(X)
        
        predictions = []
        for layer in self.output_layers:
            predictions.append(layer(self.layer_norm(hidden[-1])))
        
        if self.forecast_steps>1:
            return torch.stack(predictions, dim=1)
        else:
            return predictions[0]
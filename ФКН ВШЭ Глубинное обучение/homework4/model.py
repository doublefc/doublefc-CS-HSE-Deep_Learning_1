import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions.categorical import Categorical
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class LanguageModel(nn.Module):
    
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):

        super(LanguageModel, self).__init__()
        self.dataset = dataset  
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length


        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.rnn_type = rnn_type

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        indices = indices.to(device)
        embeds = self.embedding(indices)
        #print(embeds.shape, lengths)
        #print(packed_embeds.shape)
        outputs, hidden = self.rnn(embeds)
        #print(outputs.shape)
        logits = self.linear(outputs)
        
        
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:

        tokens = self.dataset.sp_model.encode(prefix)
        tokens = torch.tensor([self.dataset.bos_id]+tokens).unsqueeze(0).to(device)
        #print(tokens)
        
        embeds = self.embedding(tokens)
        #print(embeds.shape)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output)/temp
        #print(logits.shape)
        new_tokens = Categorical(logits=logits[:,-1:]).mode
        tokens = torch.cat([tokens, new_tokens], dim=1) 
        
        while new_tokens.item() != self.dataset.eos_id:
            embeds = self.embedding(new_tokens)
    
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output)/temp
            new_tokens = Categorical(logits=logits[:,-1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)  
            if tokens.shape[1]>=self.max_length:
                break
                
            
        generated = self.dataset.ids2text(tokens)
        return generated[0]

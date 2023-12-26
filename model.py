import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        """_summary_

        Args:
            d_model (int): size of the word embedding vector
            vocab_size (int): vocabulary size -> how many words are there in the vocabulary
        """
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self,x):
        return self.embedding(x) * torch.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        """
        Args:
            d_model (int): size of the word embedding vector
            seq_len (int): maximum length of the sentence
            dropout (_type_): to reduce overfitting
        """
        super().__init__()
        self.d_model-d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(p=dropout)

        # create a matrix of shape (seq_len,d_model)
        pe=torch.zeros(size=(seq_len, d_model))
        # create a vector of shape (seq_len, 1)
        position=torch.arange(start=0,end=seq_len, dtype=torch.float).unsqueeze(dim=1)
        denominator=torch.exp(torch.arange(start=0,end=d_model,step=2).float() * (torch.log(10000)/d_model))
        # apply the sin to even position
        pe[:, ::2]=torch.sin(input=position*denominator)
        # apply the cos to odd position
        pe[:, 1::2]=torch.cos(input=position*denominator)

        # convert dim to batch processing: (seq_len, d_model) ---> (1, seq_len, d_model)
        pe=pe.unsqueeze(dim=0)

        # save the tensor to the module but not as a learned parameter
        self.register_buffer(name="pe", tensor=pe)

    def forward(self, x):
        # note that, we don't want thos tensor to be learned as it created only once
        x=x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

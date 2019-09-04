import math

import torch
from torch.autograd import Variable
from torch.nn import Embedding, Linear, Sequential
from torch.nn.modules import (
    Dropout, LayerNorm, Module, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer
)

# Keep RNN size and dropout a constant
RNN_SIZE = 256
DROPOUT = 0.0


class Embeddings(Module):
    """
    Learned embedding to covert a vector to dimension rnn_size. The wights are
    multiplied by sqrt(rnn_size).
    """
    def __init__(self, vocab_size, rnn_size):
        super(Embeddings, self).__init__()
        self.lut = Embedding(vocab_size, rnn_size)
        self.rnn_size = rnn_size

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.rnn_size)


class PositionalEncoding(Module):
    """
    Positional encoding so that the model can make use of the order of sequence. This is
    added to the input embeddings at the bottom of the encoder and decoder stacks.

    This class is copied from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, rnn_size, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, rnn_size)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, rnn_size, 2) *
                             -(math.log(10000.0) / rnn_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Generator(Module):
    """
    Standard linear + softmax generation step.
    """
    def __init__(self, rnn_size, vocab_size):
        super(Generator, self).__init__()
        self.linear = Linear(rnn_size, vocab_size)

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.linear(x), dim=-1)


class EncoderDecoder(Module):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(
            self, encoder, decoder, generator, rnn_size, src_vocab_size, tgt_vocab_size
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

        self.src_embedding_pe = Sequential(
            Embedding(src_vocab_size, rnn_size), PositionalEncoding(rnn_size)
        )
        self.tgt_embedding_pe = Sequential(
            Embedding(tgt_vocab_size, rnn_size), PositionalEncoding(rnn_size)
        )

    def forward(
            self, src, tgt, src_mask, tgt_mask,
            src_key_padding_mask, tgt_key_padding_mask
    ):
        """
        Take in and process masked source and target sequences.

        Arguments:
            src: the sequence to the encoder; size = S x N (will be passed as S x N x E)
            tgt: the sequence to the decoder; size = T x N (will be passed as S x N x E)
            src_mask: the additive mask for the src sequence; size S x S
            tgt_mask: the additive mask for the tgt sequence; size T x T
            src_key_padding_mask: the ByteTensor mask for src keys per batch; size N x S
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch; size N x T

        where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number (must be equal to rnn_size)

        Returns:
            output: size = T x N x E

        Note:
            - src and tgt mask should be filled with float('-inf') for the masked
                positions and float(0.0) else
            - padding mask should be a ByteTensor where True values are positions that
                will be masked with float('-inf') and False values will be unchanged
        """
        memory = self.encode(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        output = self.decode(
            memory=memory, tgt=tgt, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return output

    def encode(self, src, src_mask, src_key_padding_mask):
        return self.encoder(
            src=self.src_embedding_pe(src), mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, memory, tgt, tgt_mask, tgt_key_padding_mask):
        return self.decoder(
            tgt=self.tgt_embedding_pe(tgt), memory=memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )


def build_transformer_model(
        src_vocab_size: int,
        tgt_vocab_size: int,
        rnn_size: int = RNN_SIZE,
        num_head: int = 4,
        num_layers: int = 3,
        dim_ff: int = 1024,
        dropout: float = DROPOUT
) -> EncoderDecoder:
    """
    Build transformer model based on the paper "Attention Is All You Need".

    Arguments:
         src_vocab_size: vocab size for encoder
         tgt_vocab_size: vocab size for decoder
         rnn_size: size of RNN hidden states in encoder/decoder
         num_head: the number of heads in the multi headed attention
         num_layers: number of encoder/decoder layers
         dim_ff: the dimension of the feed forward layer
         dropout: the dropout probability value
    """

    # Build encoder
    encoder_layer = TransformerEncoderLayer(rnn_size, num_head, dim_ff, dropout)
    encoder_norm = LayerNorm(rnn_size)
    encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    # Build decoder
    decoder_layer = TransformerDecoderLayer(rnn_size, num_head, dim_ff, dropout)
    decoder_norm = LayerNorm(rnn_size)
    decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)

    # Build generator
    generator = Generator(rnn_size, tgt_vocab_size)

    return EncoderDecoder(
        encoder, decoder, generator, rnn_size, src_vocab_size, tgt_vocab_size
    )


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with
    float('-inf'). Unmasked positions are filled with float(0.0).

    Example, when size = 4
    tensor([[0., -inf, -inf, -inf],
            [0.,  0.,  -inf, -inf],
            [0.,  0.,   0.,  -inf],
            [0.,  0.,   0.,   0.]])
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')
    ).masked_fill(
        mask == 1, float(0.0)
    )
    return mask


def generate_key_padding_masks(src, tgt, src_pad_index, tgt_pad_index):
    """
    Generate source and target ByteTensor masks where True values are positions that
    will be masked with float('-inf') and False values will be unchanged.
    """
    src_pad = (src == src_pad_index)
    src_mask = src_pad.bool().transpose(0, 1)

    tgt_pad = (tgt == tgt_pad_index)
    tgt_mask = tgt_pad.bool().transpose(0, 1)

    return src_mask, tgt_mask


def count_parameters(model):
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

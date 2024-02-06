"""
Adapted from The Annotated Transformer (Attention is All You Need)
https://nlp.seas.harvard.edu/annotated-transformer/
"""

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.nn.functional import log_softmax
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

### TYPES ###

Module1Input = Callable[[Tensor], Tensor]
Module4Inputs = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
Criterion = Callable[[Tensor, Tensor], Tensor]
LossLayer = Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]]


### CONSTANTS ###


TRAINING_MODES = ["train", "train+log"]


### MODULE/TENSOR FUNCTIONS ###


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce n identical layers
    """
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    attention_shape = (1, size, size)
    mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute 'Scaled Dot Product Attention'
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attention = scores.softmax(dim=-1)
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention, value), p_attention


### CLASSES ###


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % h == 0, f"d_model: {d_model}; h: {h}"
        # Assume that d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention: Optional[Tensor] = None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        query, key, value = [
            linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attention = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        del query, key, value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LayerNorm(nn.Module):
    """
    Standard LayerNorm implementation
    """

    def __init__(self, features: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.epsilon) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Module1Input) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed-forward
    """

    def __init__(
        self,
        size: int,
        self_attention: Module4Inputs,
        feed_forward: Module1Input,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of n layers
    """

    def __init__(self, layer: EncoderLayer, n: int) -> None:
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attention, source-attention, and feed-forward
    """

    def __init__(
        self,
        size: int,
        self_attention: Module4Inputs,
        source_attention: Module4Inputs,
        feed_forward: Module1Input,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attention = self_attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self, x: Tensor, memory: Tensor, source_mask: Tensor, target_mask: Tensor
    ) -> Tensor:
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.sublayers[1](
            x, lambda x: self.source_attention(x, memory, memory, source_mask)
        )
        return self.sublayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic n-layer decoder with masking
    """

    def __init__(self, layer: DecoderLayer, n: int) -> None:
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(
        self, x: Tensor, memory: Tensor, source_mask: Tensor, target_mask: Tensor
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    positional_encoding: Tensor

    def __init__(self, d_model: int, dropout: float, max_length: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        positional_encoding[:, 0::2] = torch.sin(position * denominator)
        positional_encoding[:, 1::2] = torch.cos(position * denominator)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.positional_encoding[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class GeneratorModule(nn.Module):
    """
    A standard linear + softmax generator
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(self.projection(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embed: nn.Sequential,
        target_embed: nn.Module,
        generator: GeneratorModule,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(
        self, source: Tensor, target: Tensor, source_mask: Tensor, target_mask: Tensor
    ) -> Tensor:
        """
        Take in and process masked source and target sequences.
        """
        return self.decode(
            self.encode(source, source_mask), source_mask, target, target_mask
        )

    def encode(self, source: Tensor, source_mask: Tensor) -> Tensor:
        return self.encoder(self.source_embed(source), source_mask)

    def decode(
        self, memory: Tensor, source_mask: Tensor, target: Tensor, target_mask: Tensor
    ) -> Tensor:
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)


class LabelSmoothing(nn.Module):
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist: Optional[Tensor] = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert (
            x.size(1) == self.size
        ), f"Input tensor size {x.size(1)} does not match LabelSmoothing size {self.size}"
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class Batch:
    """
    Hold a batch of data with mask during training
    """

    source: Tensor
    source_mask: Tensor
    target: Tensor
    target_y: Tensor
    target_mask: Tensor
    n_tokens: int

    def __init__(
        self, source: Tensor, target: Optional[Tensor] = None, pad: int = 2
    ) -> None:
        self.source = source
        self.source_mask = (source != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_mask(self.target, pad)
            self.n_tokens = int((self.target_y != pad).data.sum())

    @staticmethod
    def make_mask(target: Tensor, pad: int) -> Tensor:
        target_mask = (target != pad).unsqueeze(-2)
        return target_mask & subsequent_mask(target.size(-1)).type_as(target_mask.data)


@dataclass
class TrainState:
    """
    Track number of steps, examples, and tokens processed
    """

    step: int = 0
    accumulation_step: int = 0
    samples: int = 0
    tokens: int = 0


@dataclass
class SimpleLossCompute:
    generator: GeneratorModule
    criterion: Criterion

    def __call__(self, x: Tensor, y: Tensor, norm: int) -> tuple[Tensor, Tensor]:
        x = self.generator(x)
        s_loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return s_loss.data * norm, s_loss


### TRAIN/TEST FUNCTIONS ###


def make_model(
    source_vocab: int,
    target_vocab: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """
    Construct a model from hyperparameters
    """
    attention_layer = MultiHeadedAttention(h, d_model)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    positional_encoding = PositionalEncoding(d_model, dropout)
    encoder_layer = EncoderLayer(
        d_model, deepcopy(attention_layer), deepcopy(feed_forward), dropout
    )
    encoder = Encoder(encoder_layer, n)
    decoder_layer = DecoderLayer(
        d_model,
        deepcopy(attention_layer),
        deepcopy(attention_layer),
        deepcopy(feed_forward),
        dropout,
    )
    decoder = Decoder(decoder_layer, n)
    source_embed = nn.Sequential(
        Embeddings(d_model, source_vocab), deepcopy(positional_encoding)
    )
    target_embed = nn.Sequential(
        Embeddings(d_model, target_vocab), deepcopy(positional_encoding)
    )
    generator = GeneratorModule(d_model, target_vocab)
    model = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)

    # Initialize parameters with Glorot / fan average
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def generate_data(
    max_value: int, batch_size: int, n_batches: int
) -> Generator[Batch, None, None]:
    """
    Generate random data for a source-target copy task
    """
    for _ in range(n_batches):
        data = torch.randint(1, max_value, size=(batch_size, 10))
        data[:, 0] = 1
        source = data.requires_grad_(False).clone().detach()
        target = data.requires_grad_(False).clone().detach()
        yield Batch(source, target, 0)


def run_epoch(
    data_iter: Iterable[Batch],
    model: EncoderDecoder,
    loss_compute: LossLayer,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LRScheduler],
    mode: str = "train",
    accumulation_iterations: int = 1,
    train_state: TrainState = TrainState(),
) -> tuple[float, TrainState]:
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens_per_chunk = 0
    accumulation_step = 0

    for i, batch in enumerate(data_iter):
        out = model(batch.source, batch.target, batch.source_mask, batch.target_mask)
        loss, loss_node = loss_compute(out, batch.target_y, batch.n_tokens)

        if mode in TRAINING_MODES:
            assert optimizer is not None, "Cannot train without an optimizer"
            assert scheduler is not None, "Cannot train without an scheduler"

            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.source.shape[0]
            train_state.tokens += batch.n_tokens

            if i % accumulation_iterations == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_step += 1
                train_state.accumulation_step += 1

            scheduler.step()

        total_loss += int(loss)
        total_tokens += batch.n_tokens
        tokens_per_chunk += batch.n_tokens

        if i % 40 == 1 and mode in TRAINING_MODES:
            assert optimizer is not None, "Cannot train without an optimizer"

            learning_rate = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (
                    i,
                    accumulation_step,
                    loss / batch.n_tokens,
                    tokens_per_chunk / elapsed,
                    learning_rate,
                )
            )
            start = time.time()
            tokens_per_chunk = 0

        del loss, loss_node

    return total_loss / total_tokens, train_state


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    # Avoid zero raising to negative power
    if step == 0:
        step = 1
    return factor * model_size**-0.5 * min(step**-0.5, step * warmup ** (-1.5))


def greedy_decode(
    model: EncoderDecoder,
    source: Tensor,
    source_mask: Tensor,
    max_length: int,
    start_symbol: int,
) -> Tensor:
    memory = model.encode(source, source_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(source.data)
    for _ in range(max_length - 1):
        out = model.decode(
            memory, source_mask, ys, subsequent_mask(ys.size(1)).type_as(source.data)
        )
        probabilities = model.generator(out[:, -1])
        _, next_word = torch.max(probabilities, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(source.data).fill_(next_word)], dim=1
        )
    return ys


MAX_VALUE = 11
N = 2
LEARNING_RATE = 0.5
BATCH_SIZE = 80
N_EPOCHS = 20


def run() -> None:
    criterion = LabelSmoothing(MAX_VALUE, 0)
    model = make_model(MAX_VALUE, MAX_VALUE, N)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer,
        lambda step: rate(
            step, model_size=model.source_embed[0].d_model, factor=1.0, warmup=400
        ),
    )
    loss_function = SimpleLossCompute(model.generator, criterion)

    for _ in range(N_EPOCHS):
        model.train()
        run_epoch(
            generate_data(MAX_VALUE, BATCH_SIZE, 20),
            model,
            loss_function,
            optimizer,
            scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            generate_data(MAX_VALUE, BATCH_SIZE, 5),
            model,
            loss_function,
            None,
            None,
            mode="eval",
        )

    model.eval()
    source = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_length = source.shape[1]
    source_mask = torch.ones(1, 1, max_length)
    print(greedy_decode(model, source, source_mask, max_length, start_symbol=0))

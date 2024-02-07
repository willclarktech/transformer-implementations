"""
Adapted from How the RWKV language model works
https://johanwind.github.io/2023/03/23/rwkv_details.html
NOTE: Not actually a transformer
"""
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module
from torch.nn.functional import cross_entropy, one_hot, relu, sigmoid, softmax
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

torch.autograd.set_detect_anomaly(True)


argmax = torch.argmax
exp = torch.exp


class TimeMixing(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        # State
        self.last_x = torch.zeros(size, requires_grad=False)
        self.last_num = torch.zeros(size, requires_grad=False)
        self.last_den = torch.zeros(size, requires_grad=False)
        # Learned parameters
        self.decay = torch.zeros(size)
        self.bonus = torch.zeros(size)
        self.mix_k = torch.zeros(size)
        self.mix_v = torch.zeros(size)
        self.mix_r = torch.zeros(size)
        self.Wk = Linear(size, size)
        self.Wv = Linear(size, size)
        self.Wr = Linear(size, size)
        self.Wout = Linear(size, size)

    def forward(self, x: Tensor) -> Tensor:
        # Interpolate token with previous token and apply learned matrices to calculate key, value and receptance
        k = self.Wk(x * self.mix_k + self.last_x * (1 - self.mix_k))
        v = self.Wv(x * self.mix_v + self.last_x * (1 - self.mix_v))
        r = self.Wr(x * self.mix_r + self.last_x * (1 - self.mix_r))

        # RWKV attention
        exp_bonus_k = exp(self.bonus + k)
        wkv = (self.last_num + exp_bonus_k * v) / (self.last_den + exp_bonus_k)
        rwkv = sigmoid(r) * wkv

        # Final linear transformation
        y = self.Wout(rwkv)

        # Update state
        self.last_x = x.detach()
        self.num = exp(-exp(self.decay)) * self.last_num + exp_bonus_k * v
        self.den = exp(-exp(self.decay)) * self.last_den + exp_bonus_k

        return y


class ChannelMixing(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        # State
        self.last_x = torch.zeros(size, requires_grad=False)
        # Learned parameters
        self.mix_k = torch.zeros(size)
        self.mix_r = torch.zeros(size)
        self.Wk = Linear(size, size)
        self.Wr = Linear(size, size)
        self.Wv = Linear(size, size)

    def forward(self, x: Tensor) -> Tensor:
        # Interpolate token with previous token and
        k = self.Wk(x * self.mix_k + self.last_x * (1 - self.mix_k))
        r = self.Wr(x * self.mix_r + self.last_x * (1 - self.mix_r))

        # Gating
        vk = self.Wv(relu(k) ** 2)
        y = sigmoid(r) * vk

        # Update state
        self.last_x = x.detach()

        return y


class RWKV(Module):
    def __init__(self, n_layers: int, n_embed: int, size: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.embedding = Embedding(n_embed, size)
        self.layer_norms = [LayerNorm(size) for _ in range(n_layers * 2 + 2)]
        self.time_mixings = [TimeMixing(size) for _ in range(n_layers)]
        self.channel_mixings = [ChannelMixing(size) for _ in range(n_layers)]
        self.head = Linear(size, n_embed)

    def forward(self, token: Tensor) -> Tensor:
        x = self.embedding(token.squeeze())
        x = self.layer_norms[0](x)

        for i in range(self.n_layers):
            x_ = self.layer_norms[i * 2 + 1](x)
            dx = self.time_mixings[i](x_)
            x = x + dx

            x_ = self.layer_norms[i * 2 + 2](x)
            dx = self.channel_mixings[i](x_)
            x = x + dx

        x = self.layer_norms[-1](x)
        x = self.head(x)

        probs = softmax(x, dim=-1)

        return probs


class RandomSequenceDataset(Dataset):
    def __init__(self, sequence_length: int, num_sequences: int) -> None:
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, _: int) -> tuple[Tensor, Tensor]:
        sequence = torch.randint(0, 10, (self.sequence_length,))
        return sequence[:-1], sequence[1:]


def greedy_decode(
    model: RWKV,
    source: [int],
    max_value: int,
) -> Tensor:
    ys = []
    for token in source:
        probs = model(
            # NOTE: Pylint does not recognize one_hot as a function for some reason
            # pylint: disable=not-callable
            one_hot(
                torch.scalar_tensor(token, dtype=torch.int64).squeeze(), max_value
            ).squeeze()
        )
        _, next_token = torch.max(probs, dim=1)
        next_token = next_token.data[0]
        y = argmax(next_token)
        ys.append(int(y))
    return ys


def run_epoch(
    model: RWKV,
    optimizer: Optimizer,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    max_value: int,
) -> float:
    total_loss: float = 0
    for input_seq, target_seq in data_loader:
        optimizer.zero_grad()
        loss = 0
        for i in range(len(input_seq) - 1):
            input_token = input_seq[i]
            target_token = target_seq[i + 1]

            # NOTE: Pylint does not recognize one_hot as a function for some reason
            # pylint: disable=not-callable
            input_tensor = one_hot(
                input_token.clone().detach(),
                num_classes=max_value,
            )
            # NOTE: Pylint does not recognize one_hot as a function for some reason
            # pylint: disable=not-callable
            target_tensor = one_hot(
                target_token.clone().detach(),
                num_classes=max_value,
            )

            logits = model(input_tensor)
            loss += cross_entropy(logits.view(-1, max_value), target_tensor.view(-1))

        loss /= len(input_seq) - 1
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


N_LAYERS = 24
SIZE = 4
MAX_VALUE = 10
SEQUENCE_LENGTH = 10
N_SEQUENCES = 80
BATCH_SIZE = 32
LEARNING_RATE = 1
N_EPOCHS = 20


def run() -> None:
    model = RWKV(N_LAYERS, MAX_VALUE, SIZE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    dataset = RandomSequenceDataset(SEQUENCE_LENGTH, N_SEQUENCES)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(N_EPOCHS):
        model.train()
        loss = run_epoch(model, optimizer, data_loader, MAX_VALUE)
        average_loss = loss / len(data_loader)
        print(f"Epoch {epoch}: average loss = {average_loss}")

    model.eval()
    source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    output = greedy_decode(model, source, MAX_VALUE)
    print(f"Source: {source}")
    print(f"Output: {output}")

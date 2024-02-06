"""
Adapted from How the RWKV language model works
https://johanwind.github.io/2023/03/23/rwkv_details.html
NOTE: Not actually a transformer
"""
import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module
from torch.nn.functional import one_hot, relu, sigmoid, softmax

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
        self.last_x = x
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
        self.last_x = x

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
            x += dx

            x_ = self.layer_norms[i * 2 + 2](x)
            dx = self.channel_mixings[i](x_)
            x += dx

        x = self.layer_norms[-1](x)
        x = self.head(x)

        probs = softmax(x, dim=-1)

        return probs


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


N_LAYERS = 24
N_EMBED = 10
SIZE = 4
MAX_VALUE = 10


def run() -> None:
    model = RWKV(N_LAYERS, N_EMBED, SIZE)
    source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    output = greedy_decode(model, source, MAX_VALUE)
    print(f"Source: {source}")
    print(f"Output: {output}")

import math

import torch


#########################
# Singleton Seed Sampler
#########################


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Manager(metaclass=Singleton):
    def init_iter(self):
        self.iter = 0

    def update_iter(self):
        self.iter += 1

    def get_iter(self):
        return self.iter


class SeedSetter(metaclass=Singleton):
    """
    Constant sampler - always returns seed=current update step
    The seed is updated externally from the training loop.
    """

    def __init__(self):
        self.current_rng_state = None
        self.g = None

    def set_seed(self, seed, device):
        self.current_rng_state = torch.get_rng_state()
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)

    def reset_seed(self):
        assert self.current_rng_state is not None, "SeedSetter has not been initialized with set_seed."
        torch.set_rng_state(self.current_rng_state)

    def get_generator(self):
        assert self.g is not None, "SeedSetter's generator is not initialized. Call set_seed first."
        return self.g


def rng():
    return SeedSetter().get_generator()


############################
# Random Sampling Functions
############################


def sample_gaussian_sketch(n, r, device, dtype, b=None):
    r = round(n * r) if r <= 1 else int(r)
    if b is None:
        mat = torch.randn(n, r, device=device, dtype=dtype, generator=rng())
    else:
        mat = torch.randn(b, n, r, device=device, dtype=dtype, generator=rng())
    mat *= (1 / math.sqrt(r))
    return mat


def sample_sketch(seed, proj_type, *args):
    SeedSetter().set_seed(seed, device=args[-2])
    if proj_type == 'gaussian':
        out = sample_gaussian_sketch(*args)
    else:
        raise ValueError('Unsupported proj_type', proj_type)
    SeedSetter().reset_seed()
    return out


def sample_velora(input, rank):
    """
    input should be reshaped from (b, l, n) to (b, l, r, n/r) and mapped to (b, l, r, 1)
    So - projection should be of shape (n/r, 1)

    a.k.a - reshaping input and mean on first dimensions.
    """
    b, l, n = input.shape
    small_n = round(n * rank)

    proj = input.reshape(b, l, small_n, -1).flatten(end_dim=-2).mean(dim=0, keepdim=True).T
    return proj


def get_svd_matrix(x, rank, full_matrices):
    x = x.view(-1, x.size()[-1])  # bld to (bl,d)
    bl, n = x.shape
    rank_n = round(n * rank) if rank <= 1 else int(rank)
    U, S, Vh = torch.linalg.svd(x.to(torch.float), full_matrices=full_matrices)

    out = Vh[:rank_n, :].to(device=x.device, dtype=x.dtype).T.detach()  # shape: r,d

    del U, S, Vh
    return out


##################
# Projector Class
##################

class Projector:
    def __init__(self, rank, proj_type, scale, gap, full_matrices, single_gpu=True):
        self.rank = rank
        self.proj_type = proj_type
        self.scale = scale
        self.seed = torch.randint(1, int(1e6), (1,)).item()
        self.gap = gap
        self.full_matrices = full_matrices
        self.last_update = None
        self.proj_right = None
        self.single_gpu = single_gpu

    def project(self, x):
        """
        x is of size: (B, L, N)
        saves for backward (r1(bl),n) if left, (b,l,r2(n)) if right, and (r1(bl),r2(bl)) if full
        """
        if self.proj_type == 'id':
            return x
        if not hasattr(self, "n"):
            if len(x.shape) == 3:
                self.b, self.seq_ln, self.n = x.shape
            else:  # weight in multi linear
                _, self.n = x.shape

        iter = Manager().get_iter()
        if self.proj_type == "gaussian":

            # check if the seed can be outdated:
            if (self.last_update is None) or (self.last_update < iter):
                if (self.gap is not None) and ((iter % self.gap) == 0):
                    self.seed += 1
                    self.last_update = iter

            # sample sketch
            proj_right = sample_sketch(self.seed, self.proj_type, self.n, self.rank, x.device, x.dtype)

        else:
            assert "svd" in self.proj_type or 'velora' in self.proj_type  # ["svd", "svd_sum", "svd_avg", "svd_single"]
            if (self.gap is not None) and ((iter % self.gap) == 0):
                # check if projection can be outdated:
                if (self.last_update is None) or (self.last_update < iter):
                    if 'svd' in self.proj_type:
                        self.proj_right = get_svd_matrix(x, self.rank, self.full_matrices)
                    elif 'velora' in self.proj_type:
                        self.proj_right = sample_velora(x, self.rank)
                    self.last_update = iter

            proj_right = self.proj_right

        if 'velora' in self.proj_type:
            b, l, n = x.shape
            small_n = round(n * self.rank) if self.rank <= 1 else int(self.rank)
            x = (x.reshape(b, l, small_n, -1) @ proj_right).squeeze(-1)
        else:
            x = x @ proj_right

        del proj_right
        return x

    def project_back(self, grad):
        """
        grad is of shape:  (*dims, r(n))
        returns *dims, n and multiplies by galore's alpha scale
        """
        if self.proj_type == 'id':
            return grad * self.scale
        elif self.proj_type == "gaussian":
            proj_right = sample_sketch(self.seed, self.proj_type, self.n, self.rank,
                                       grad.device, grad.dtype)
        elif self.proj_type == 'velora':
            return (grad.unsqueeze(-1) @ self.proj_right.T).flatten(start_dim=-2) * self.scale

        else:
            proj_right = self.proj_right

        if proj_right is not None:
            return (grad @ proj_right.T) * self.scale

        del proj_right
        return grad

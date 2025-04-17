import torch

units_dict = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

def obj_to_bytes(obj):
    return obj.untyped_storage().nbytes()


def llama_params(config, fp16, unit="MB"):
    """
    right now its just for a single decoder
    """
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    n_heads = config.num_attention_heads
    max_position_embeddings = config.max_position_embeddings
    layers = config.num_hidden_layers

    u = units_dict[unit]
    memory = 0
    bytes = 2 if fp16 else 4

    # RMS (2 layers)
    memory += 2*hidden_size*bytes

    # MLP
    memory += 3*hidden_size*intermediate_size*bytes

    # ATTENTION
    memory +=  4*hidden_size*hidden_size*bytes

    # ROTARY BUFFERS
    memory += 2*(hidden_size/n_heads)*max_position_embeddings*bytes

    memory *= layers

    return memory/u



def rms_memory(b, l, d,  fp16, intermediate=True, units="MB"):
    """
    Without Type Conversion: input is fp32
    ------------------------
    For first layer:
    ################
    * input not saved
    * only hidden_states multiplied by weight is saved

    For Intermediate Layer:
    ########################
    * original input is saved   (fp32)
    * root square variance is saved (fp32)
    * final hidden states multiplied by weight is saved (fp32)


    With Type Conversion:
    ---------------------
    For first layer:  input is fp16
    ################
    * input not saved
    * only hidden_states multiplied by weight is saved (fp16)

    For Intermediate Layer:
    ########################
    * original input is saved  (fp16)
    * input after conversion is saved (fp32)
    * root square variance is saved (fp32)
    * final hidden states multiplied by weight is saved   (fp16)
    """
    u = units_dict[units]

    memory = 0
    x_in = torch.zeros(b, l, d, dtype=(torch.float16 if fp16 else torch.float32))
    # x_in = torch.zeros(b, l, d, dtype=torch.float32)
    in_size = obj_to_bytes(x_in)/u

    if intermediate:
        # save original input in same precision
        memory += in_size

        if fp16:
            # if we convert, input is also saved in full precision
            memory += 2*in_size

        # sqrt(var + eps) always saved in full precision
        memory +=  obj_to_bytes(torch.zeros(b, l, dtype=torch.float32))/u
        # memory +=  obj_to_bytes(torch.zeros(b, l, dtype=torch.float16))/u

    # save output in save shape\precision as input
    memory += in_size

    return memory



def mlp_memory(b, l, hidden_size, intermediate_size, fp16, efficient_linear=False, rank=1, multi=False, units="MB"):
    """
    out = down( act_fn(gate(x)) * up(x) )

    * original input is saved as activation (gate & up act)

    * gate output is saved (silu act)

    * up output is saved (hadamard act)
    * silu output is saved (hadamard act)
    * hadamard output is saved (down act)
    """
    u = units_dict[units]

    memory_linear = memory_hadamard = memory_act = 0

    x_in = torch.zeros(b, l, hidden_size, dtype=(torch.float16 if fp16 else torch.float32))
    in_size = obj_to_bytes(x_in) / u

    x_inter = torch.zeros(b, l, intermediate_size, dtype=(torch.float16 if fp16 else torch.float32))
    inter_size = obj_to_bytes(x_inter) / u

    # up and gate activation
    if not efficient_linear:
        memory_linear += in_size
    elif multi:
        memory_linear += in_size*rank
    else:
        memory_linear += 2*in_size*rank

    # silu activation
    memory_act += inter_size

    # hadamard activation
    memory_hadamard += 2 * inter_size

    # down activation
    if not efficient_linear:
        memory_linear += inter_size
    else:
        memory_linear += in_size * rank

    return memory_linear, memory_hadamard, memory_act


def attention_memory(b, l, d, n_heads, fp16, efficient_linear=False, rank=1, multi=False, units="MB"):
    """
    * original input is saved            of size (b,l,d)        (q, k, v activation)
    * q_proj output is not saved
    * k_proj output is not saved
    * values is saved                    of size (b,l,d)        (flash_attn act)
    * rotary out: q values is saved      of size (b,l,d)        (flash_attn act)
    * rotary out: k values is saved      of size (b,l,d)        (flash_attn act)
    * softmax norm statistics, 2 tensors of size (b,l,n_head)   (flash_attn act)
    * scaled_dot_prod out?? is saved     of size (b,l,d)        (flash_attn & o activation)
    """
    u = units_dict[units]

    memory_linear = memory_flash_attn = 0

    x_in = torch.zeros(b, l, d, dtype=(torch.float16 if fp16 else torch.float32))
    in_size = obj_to_bytes(x_in) / u

    flash_attn_aux = torch.zeros(b, l, n_heads, dtype=(torch.float16 if fp16 else torch.float32))
    flash_attn_aux = obj_to_bytes(flash_attn_aux) / u

    # save activation for k, q, v
    if not efficient_linear:
        memory_linear += in_size
    elif multi:
        memory_linear += in_size*rank
    else:
        memory_linear += 3*in_size*rank

    # save activations for flash attn
    memory_flash_attn += 3*in_size
    memory_flash_attn += 2*flash_attn_aux

    # save activation for o & flash_attn
    memory_flash_attn += in_size

    return memory_linear, memory_flash_attn


def decoder_block_memory(b, l, config, fp16, efficient_linear=False, rank=1, multi=False):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    n_head = config.num_attention_heads

    norm_memory = 2*rms_memory(b, l, hidden_size, fp16=fp16)
    memory_linear_mlp, memory_hadamard, memory_act = mlp_memory(b, l, hidden_size, intermediate_size, fp16=fp16,
                                                                efficient_linear=efficient_linear, rank=rank,
                                                                multi=multi)
    memory_linear_attn, memory_flash_attn = attention_memory(b, l, hidden_size, n_head, fp16=fp16,
                                                             efficient_linear=efficient_linear, rank=rank,
                                                             multi=multi)

    return norm_memory, memory_linear_mlp, memory_hadamard, memory_act, memory_linear_attn, memory_flash_attn



def embedding_memory(b, l, d, fp16, units="MB"):
    """
    * original input (int64) is saved size (b, l)
    """


    pass
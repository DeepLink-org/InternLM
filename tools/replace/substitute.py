import torch
import torch_dipu
def apply_rotary(x1, x2, cos, sin, conj):
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    out1 = out1.to(torch.float16)
    out2 = out2.to(torch.float16)
    return out1, out2

# def apply_rotary(x1, x2, cos, sin, conj):
#     if not conj:
#         out1 = x1 * cos - x2 * sin
#         out2 = x1 * sin + x2 * cos
#     else:
#         out1 = x1 * cos + x2 * sin
#         out2 = -x1 * sin + x2 * cos
#     return out1, out2


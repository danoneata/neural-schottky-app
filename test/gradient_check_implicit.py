import pdb
import sys

import torch
from torch.autograd import gradcheck
from torch.autograd.functional import jacobian

sys.path.append('.')
from core import SolveExpLinear, SolveExpLinearImplicit

inp = torch.randn(100).double()
inp = inp * 1e3
inp = inp.requires_grad_()
model1 = SolveExpLinear().double()
model2 = SolveExpLinearImplicit().double()
out1 = model1(inp)
out2 = model2(inp)
# pdb.set_trace()
print(gradcheck(model1, inp, eps=1e-5, atol=1e-3, check_undefined_grad=False))
print(gradcheck(model2, inp, eps=1e-5, atol=1e-3, check_undefined_grad=False))

J1 = torch.diag(jacobian(model1, inp))
J2 = torch.diag(jacobian(model2, inp))
print(J1)
print(J2)
print(J1 - J2)
# pdb.set_trace()
from dataclasses import dataclass

import numpy as np
import streamlit as st
import torch

from torch import nn
from torch import optim

from plot import show_progress
from constants import An, compute_As, ureg, K_DIV_Q


# Constants
MAX_ITER = 1_000


def fwd_solver_torch(f, z_init, max_iter=100, **kwargs):
    i = 0
    z_prev, z = z_init, f(z_init, **kwargs)
    while not torch.all(torch.abs(z_prev - z) <= 1e-8) and i <= max_iter:
        z_prev, z = z, f(z, **kwargs)
        i += 1
    return z


class DiodeNetSolve(nn.Module):
    """Predict fixed-point solution of the diode equation."""

    def __init__(self, *, Φ, peff, rs_net, n_net, As, An):
        super().__init__()
        # parameters
        self.Φ = Φ
        self.peff = nn.Parameter(torch.tensor(peff))
        # networks
        self.rs_net = rs_net
        self.n_net = n_net
        # fixed values
        self.As = As
        self.An = An

    def get_peff(self):
        return torch.clamp(self.peff, min=0)

    def forward(self, *, V, T, max_iter=10, **kwargs):
        n = self.n_net(T)
        Vth = K_DIV_Q * T
        Is = self.predict_Is(V=V, T=T)
        Rs = self.rs_net(T)

        nVth = n * Vth
        log_term = torch.log(Rs) + torch.log(Is) - torch.log(nVth)
        x = (V + Rs * Is) / nVth + log_term

        def f(y, x):
            ey = torch.exp(y)
            ey1 = ey + 1
            yeyx = y + ey - x
            numer = 2 * yeyx * ey1
            denom = 2 * ey1**2 - yeyx * ey
            return y - numer / denom

        def g(y, x):
            # avoid overflow for large values of x
            ey = torch.exp(-y)
            ey1 = ey + 1
            yey1xey = (y - x) * ey + 1
            numer = 2 * yey1xey * ey1
            denom = 2 * ey1**2 - yey1xey
            return y - numer / denom

        e = np.exp(1)
        idxs = x > e
        y_init = x.clone()
        y_init[idxs] = torch.log(y_init[idxs])

        idxs1 = x < 0
        idxs2 = ~idxs1
        y_star = torch.zeros_like(y_init)
        y_star[idxs1] = fwd_solver_torch(
            f, y_init[idxs1], max_iter=max_iter, x=x[idxs1]
        )
        y_star[idxs2] = fwd_solver_torch(
            g, y_init[idxs2], max_iter=max_iter, x=x[idxs2]
        )

        I_star = (V - nVth * (y_star - log_term)) / Rs

        return I_star

    def predict_I(self, *, I, V, T):
        TK = T
        Vth = K_DIV_Q * TK  # V
        Is = self.predict_Is(V=V, T=T)
        Rs = self.rs_net(T)
        I = Is * (torch.exp((V - Rs * I) / (self.n_net(T) * Vth)) - 1)
        return torch.clamp(I, min=1e-9)

    def log_Is(self, *, V, T):
        TK = T
        Vth = K_DIV_Q * TK  # V
        Φ = self.Φ()
        peff = self.get_peff()
        An = torch.tensor(float(self.An))
        As = torch.tensor(float(self.As))
        return torch.log(An) + torch.log(As) + 2 * torch.log(TK) - Φ / Vth - peff

    def predict_Is(self, *, V, T):
        TK = T
        Vth = K_DIV_Q * TK  # V
        Φ = self.Φ()
        peff = self.get_peff()
        return self.An * self.As * TK**2 * torch.exp(-Φ / Vth - peff)


class DiodeMixtureNet(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, *, V, T):
        I_pred = torch.stack([net(V=V, T=T, max_iter=MAX_ITER) for net in self.nets])
        return I_pred.sum(dim=0)


class PhiSigmoid(nn.Module):
    def __init__(self, p):
        super(PhiSigmoid, self).__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self):
        return 2.0 * torch.sigmoid(self.p)

    def __str__(self):
        value = self.forward()
        return f"{value:.4f}"


class RsBias(nn.Module):
    def __init__(self, temps, r):
        super(RsBias, self).__init__()
        self.temps = temps
        self.temp_to_idx = {t: i for i, t in enumerate(temps)}
        num_temps = len(temps)
        if isinstance(r, float):
            rs = torch.tensor([r]).repeat(num_temps)
        else:
            rs = torch.tensor(r)
        self.rs = nn.Parameter(rs)

    def forward(self, temps):
        if temps.dim() == 0:
            idx = self.temp_to_idx[temps.item()]
            return torch.clamp(self.rs[idx], min=0.0)
        else:
            idxs = torch.tensor([self.temp_to_idx[temp.item()] for temp in temps])
            return torch.clamp(self.rs[idxs], min=0.0)

    def __str__(self):
        return " ".join("{:.1f}".format(torch.clamp(r, min=0.0)) for r in self.rs)


@dataclass
class NFixed:
    n: float

    def __call__(self, *args, **kwargs):
        return self.n

    def __str__(self):
        return str(self.n)


def compute_r2(true, pred):
    μ = true.mean()
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - μ) ** 2).sum()
    return 1 - ss_res / ss_tot


def create_net(diameter, temps, params_init, to_freeze=tuple()):
    D = diameter * ureg.micrometers
    As = compute_As(D.to("cm").magnitude)
    phi_logit = torch.logit(torch.tensor(params_init.Φ / 2))
    rs = [params_init.rs[t] for t in temps]
    diode = DiodeNetSolve(
        Φ=PhiSigmoid(phi_logit),
        peff=params_init.peff,
        rs_net=RsBias(temps=temps, r=rs),
        n_net=NFixed(n=1.03),
        As=As,
        An=An,
    )
    for name, param in diode.named_parameters():
        if name.split(".")[0] in to_freeze:
            param.requires_grad = False
    return diode


def create_mixture_net(diameter, temps, params_init_all):
    nets = [create_net(diameter, temps, *p) for p in params_init_all]
    return DiodeMixtureNet(nets)


def predict_net(mixture, data):
    mixture.eval()
    V = torch.tensor(data["V"].to_numpy())
    T = torch.tensor(data["T"].to_numpy())
    Istar = torch.stack(
        [net.forward(V=V, T=T, max_iter=MAX_ITER) for net in mixture.nets]
    )
    return Istar.detach().numpy()


def fit(mixture, data, container, num_steps=100, plot_every_n_steps=10, *, lr):
    optimizer = optim.LBFGS(
        mixture.parameters(),
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )
    SS = 4
    mixture.train()

    for i in range(num_steps):
        data_ss = data
        T = torch.tensor(data_ss["T"].to_numpy())
        V = torch.tensor(data_ss["V"].to_numpy())
        I = torch.tensor(data_ss["I"].to_numpy())

        def closure():
            mixture.zero_grad()
            I_pred = mixture(V=V, T=T)

            log_I_true = torch.log10(I)
            log_I_pred = torch.log10(I_pred)

            loss = -compute_r2(log_I_true, log_I_pred)
            loss.backward()

            if torch.isnan(loss):
                raise ValueError("Fitting diverged")

            # print("loss: {:.9f}".format(loss.item()))
            return loss

        optimizer.step(closure)
        st.session_state.mixture_net = mixture
        st.session_state.iter = i

        with container.container():
            to_plot = (i + 1) % plot_every_n_steps == 0
            st.markdown(f"## Step: {i + 1}")
            I_pred_all = predict_net(mixture, data[::SS])
            show_progress(mixture, data[::SS], I_pred_all, to_plot=to_plot)
            mixture.train()

    # return mixture

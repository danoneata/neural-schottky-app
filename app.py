import numpy as np
import streamlit as st

from core import create_mixture_net, predict_net, fit
from data import DATASETS, Parameters
from plot import show_progress


def set_diode_menu(i, params_init_0):
    st.markdown(f"## Diode {i + 1}")
    Φ0 = max(0.0, params_init_0.Φ - 0.1 * i)
    Φ = st.number_input(
        "Φ",
        value=Φ0,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        key=f"Φ{i}",
    )
    peff0 = params_init_0.peff + i
    peff = st.number_input(
        "p_eff",
        value=peff0,
        min_value=0.0,
        step=0.05,
        key=f"peff{i}",
    )
    rs0 = params_init_0.rs[0] + 10**i
    rs = st.number_input(
        "Rs",
        value=rs0,
        min_value=0.0,
        step=1.0,
        key=f"Rs{i}",
    )
    return Parameters(Φ=Φ, peff=peff, rs=[rs])


def main():
    was_clicked = False

    with st.sidebar:
        st.markdown("# Dataset")
        dataset_name = st.selectbox("Select dataset", DATASETS.keys())
        dataset = DATASETS[dataset_name]
        temperatures = st.multiselect(
            "Select temperatures",
            dataset.get_available_temps(),
            dataset.temps_default,
        )
        st.markdown("---")

        st.markdown("# Parameters")
        num_diodes = st.number_input("Number of diodes", min_value=1, value=1, step=1)

        params_init_all = [
            set_diode_menu(i, dataset.params_init) for i in range(num_diodes)
        ]

        st.markdown("---")
        st.markdown("# Fitting")
        num_steps = st.number_input("Number of steps", min_value=1, value=100, step=5)
        plot_every_n_steps = st.number_input(
            "How often to plot", min_value=1, value=10, step=5,
            help="Generates plots every n steps.",
        )
        lr = st.number_input("Learning rate", min_value=1e-5, value=0.04, step=0.01)

    if len(temperatures) == 0:
        st.warning("You need to select a temperature.")
        return

    data = dataset.load(temperatures)

    SS = 4
    temps_np = data["T"].values
    temps_np = np.sort(np.unique(temps_np))

    mixture_net = create_mixture_net(
        dataset,
        temps_np,
        params_init_all,
    )

    if "mixture_net" not in st.session_state:
        st.session_state.mixture_net = mixture_net
        st.session_state.iter = iter

    container_fit = st.empty()

    with st.sidebar:
        was_clicked = st.button(
            "Fit",
            on_click=fit,
            args=(mixture_net, data, container_fit),
            kwargs={
                "num_steps": num_steps,
                "plot_every_n_steps": plot_every_n_steps,
                "lr": lr,
            },
        )

    if was_clicked:
        I_pred_all = predict_net(st.session_state.mixture_net, data[::SS])
        show_progress(st.session_state.mixture_net, data[::SS], I_pred_all)
    else:
        I_pred_all = predict_net(mixture_net, data[::SS])
        show_progress(mixture_net, data[::SS], I_pred_all)


if __name__ == "__main__":
    main()

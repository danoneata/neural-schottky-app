import pdb

import pandas as pd
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
        format="%.3f",
        key=f"Φ{i}",
    )
    peff0 = params_init_0.peff + i
    peff = st.number_input(
        "p_eff",
        value=peff0,
        min_value=0.0,
        step=0.05,
        format="%.3f",
        key=f"peff{i}",
    )
    rs0 = params_init_0.rs[0] + 10**i
    rs = st.number_input(
        "Rs",
        value=rs0,
        min_value=0.0,
        step=1.0,
        format="%.1f",
        key=f"Rs{i}",
    )
    return Parameters(Φ=Φ, peff=peff, rs=[rs])


PARAMS_INIT = Parameters(Φ=1.3, peff=0.29, rs=[50.0])


def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    min_I = 1e-9

    data = pd.read_csv(uploaded_file, index_col=0)
    data = data.unstack().reset_index()
    data.columns = ["T", "V", "I"]
    data = data.astype(float)
    # data["T"] = data["TK"] - 273.15

    data = data.dropna(axis=0, how="any")
    data = data[data["I"] >= min_I]

    return data


def get_available_temps(data):
    if data is None:
        return []
    return sorted(data["T"].unique().tolist())


def get_temps_default(data):
    if data is None:
        return []
    temps = get_available_temps(data)
    return temps[::2]


def main():
    was_clicked = False

    with st.sidebar:
        st.markdown("# Dataset")
        uploaded_file = st.file_uploader(
            "Upload dataset",
            type="csv",
            help=(
                "Assumes the following format: "
                "first line are the temperatures (in K), "
                "first column are the voltages (in V), and "
                "the rest are the currents (in A)."
                "The values should be comma-separated."
                "Missing values should be empty."
            ),
        )
        diameter = st.number_input(
            "Diode's diameter (µm)",
            value=400.0,
            step=10.0,
            min_value=0.0,
            format="%f",
        )

        data = load_data(uploaded_file)

        temperatures = st.multiselect(
            "Select temperatures",
            get_available_temps(data),
            get_temps_default(data),
        )
        st.markdown("Voltage range")
        cols = st.columns(2)
        Vmin = cols[0].number_input("V min", min_value=0.0, value=1e-9, step=0.1, format="%g")
        Vmax = cols[1].number_input("V max", min_value=Vmin, value=2.5, step=0.1)
        st.markdown("---")

        st.markdown("# Parameters")
        num_diodes = st.number_input("Number of diodes", min_value=1, value=1, step=1)

        params_init_all = [set_diode_menu(i, PARAMS_INIT) for i in range(num_diodes)]

        st.markdown("---")
        st.markdown("# Fitting")
        num_steps = st.number_input("Number of steps", min_value=1, value=100, step=5)
        plot_every_n_steps = st.number_input(
            "How often to plot",
            min_value=1,
            value=10,
            step=5,
            help="Generates plots every n steps.",
        )
        lr = st.number_input("Learning rate", min_value=1e-5, value=0.04, step=0.01)

    if data is not None:
        st.success("Dataset uploaded successfully!")
    else:
        st.warning("No dataset uploaded.")
        st.stop()

    if len(temperatures) == 0:
        st.warning("You need to select a temperature.")
        st.stop()

    # st.dataframe(data)

    SS = 4
    data = data[data["V"] >= Vmin]
    data = data[data["V"] <= Vmax]
    data = data[data["T"].isin(temperatures)]

    mixture_net = create_mixture_net(
        diameter,
        temperatures,
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

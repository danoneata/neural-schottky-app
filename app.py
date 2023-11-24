import pdb

import pandas as pd
import numpy as np
import streamlit as st

from toolz import first

from core import create_mixture_net, predict_net, fit
from data import DATASETS, Parameters
from plot import show_progress


def set_diode_menu(i, params_init, temperatures):
    to_freeze = []

    st.markdown(f"## Diode {i + 1}")
    cols = st.columns([0.7, 0.3])
    Φ = cols[0].number_input(
        "Φ (V)",
        value=params_init.Φ,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        format="%.3f",
        key=f"Φ/{i}",
    )

    Φ_frozen = cols[1].toggle("🔒", key=f"Φ-frozen/{i}")
    if Φ_frozen:
        to_freeze.append("Φ")

    cols = st.columns([0.7, 0.3])
    peff = cols[0].number_input(
        "p_eff",
        value=params_init.peff,
        min_value=0.0,
        step=0.05,
        format="%.3f",
        key=f"peff={i}",
    )

    peff_frozen = cols[1].toggle("🔒", key=f"peff-frozen/{i}")
    if peff_frozen:
        to_freeze.append("peff")

    if len(params_init.rs) in {0, 1}:
        index = 0
    else:
        index = 1

    cols = st.columns([0.7, 0.3])
    rs_type = cols[0].radio(
        "Rs initialisation type",
        ("Same for all temperatures", "Individual per temperature"),
        index=index,
        key=f"rs_type/{i}",
    )

    rs_frozen = cols[1].toggle("🔒", key=f"rs-frozen/{i}")
    if rs_frozen:
        to_freeze.append("rs_net")

    if rs_type == "Same for all temperatures":
        try:
            rs0 = first(params_init.rs.values())
        except StopIteration:
            rs0 = DEFAULT_RS
        rs = st.number_input(
            "Rs (Ω)",
            value=rs0,
            min_value=0.0,
            step=1.0,
            format="%.1f",
            key=f"Rs={i}",
        )
        rs = {t: rs for t in temperatures}
    else:
        with st.expander("Values"):
            rs = {t: 0.0 for t in temperatures}
            for j, t in enumerate(temperatures):
                try:
                    rs0 = params_init.rs[t]
                except KeyError:
                    t_closest = min(params_init.rs.keys(), key=lambda x: abs(x - t))
                    rs0 = params_init.rs[t_closest]
                rs[t] = st.number_input(
                    "Rs (Ω) at {:.1f} K".format(t),
                    value=rs0,
                    min_value=0.0,
                    step=1.0,
                    format="%.1f",
                    key=f"Rs={i},{j}",
                )

    return Parameters(Φ=Φ, peff=peff, rs=rs), tuple(to_freeze)


DEFAULT_Φ = 1.3
DEFAULT_PEFF = 0.29
DEFAULT_RS = 50.0


def get_params_init(temperatures, i):
    return Parameters(
        Φ=max(0.0, DEFAULT_Φ - 0.1 * i),
        peff=DEFAULT_PEFF + i,
        rs={t: DEFAULT_RS + 10**i for t in temperatures},
    )


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


# @st.cache(suppress_st_warning=True)
def show(data, mixture_net):
    SS = 4
    I_pred_all = predict_net(mixture_net, data[::SS])
    show_progress(mixture_net, data[::SS], I_pred_all)


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
            "Temperatures",
            get_available_temps(data),
            get_temps_default(data),
        )
        temperatures = sorted(temperatures)

        st.markdown("Voltage range")
        cols = st.columns(2)
        Vmin = cols[0].number_input(
            "Min (V)", min_value=0.0, value=1e-9, step=0.001, format="%g"
        )
        Vmax = cols[1].number_input("Max (V)", min_value=Vmin, value=2.5, step=0.1)
        st.markdown("---")

        st.markdown("# Parameters")
        num_diodes = st.number_input("Number of diodes", min_value=1, value=1, step=1)

        if "mixture_net" in st.session_state:

            mixture_net = st.session_state["mixture_net"]

            def get_params(i):
                if i < len(mixture_net.nets):
                    diode = mixture_net.nets[i]
                    ts = diode.rs_net.temps
                    rs = diode.rs_net.rs.detach().tolist()
                    rs_dict = {t: r for t, r in zip(ts, rs)}
                    return Parameters(
                        Φ=diode.Φ().detach().item(),
                        peff=diode.get_peff().item(),
                        rs=rs_dict,
                    )
                else:
                    return get_params_init(temperatures, i)

            params_init = [get_params(i) for i in range(num_diodes)]

        else:
            params_init = [get_params_init(temperatures, i) for i in range(num_diodes)]

        # with st.form("parameters"):
        params_init_all = [
            set_diode_menu(i, params_init[i], temperatures) for i in range(num_diodes)
        ]
        # submitted_params_init = st.form_submit_button("Submit")

        st.markdown("---")
        st.markdown("# Fitting")
        num_steps = st.number_input("Number of steps", min_value=1, value=10, step=5)
        plot_every_n_steps = st.number_input(
            "How often to plot",
            min_value=1,
            value=10,
            step=5,
            help="Generates plots every n steps.",
        )
        lr = st.number_input("Learning rate", min_value=1e-5, value=0.04, step=0.01)
        differentiation_type = st.selectbox("Type of differentiation", ["unrolled", "implicit"])

    if data is not None:
        # st.success("Dataset uploaded successfully!")
        pass
    else:
        st.warning("No dataset uploaded.")
        st.stop()

    # if not submitted_params_init:
    #     st.warning("No parameters submitted.")
    #     st.stop()

    if len(temperatures) == 0:
        st.warning("You need to select a temperature.")
        st.stop()

    # st.dataframe(data)

    data = data[data["V"] >= Vmin]
    data = data[data["V"] <= Vmax]
    data = data[data["T"].isin(temperatures)]

    mixture_net = create_mixture_net(
        diameter,
        temperatures,
        differentiation_type,
        params_init_all,
    )

    if "mixture_net" not in st.session_state:
        st.session_state.mixture_net = mixture_net
        st.session_state.iter = iter
        st.session_state.time_spent = 0.0

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
        st.markdown("Time spent: {:.3}s".format(st.session_state.time_spent))
        show(data, st.session_state.mixture_net)
    else:
        show(data, mixture_net)


if __name__ == "__main__":
    main()

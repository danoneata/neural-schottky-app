from matplotlib import pyplot as plt
from sklearn import metrics

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

sns.set_theme()


def plot_data(df):
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="V",
        y="I",
        hue="T",
        palette="tab20c",
        ax=ax,
        legend=False,
    )
    ax.set(yscale="log")
    return fig, ax


def show_progress(mixture, data_orig, I_pred_all, to_plot=True):
    def plot(data, col=st):
        fig, ax = plt.subplots()
        ax = sns.scatterplot(
            data=data,
            x="V",
            y="I",
            hue="T",
            palette="tab20c",
            ax=ax,
            legend=False,
        )
        ax = sns.lineplot(
            data=data,
            x="V",
            y="I-pred",
            hue="T",
            palette="tab20c",
            ax=ax,
            legend=False,
        )
        ax.set_ylim([1e-10, 1])
        ax.set(yscale="log")
        fig.tight_layout()
        col.pyplot(fig)

    def plot_independent(data, I_pred, col):
        data1 = pd.concat([data["V"], I_pred], axis=1)
        data1 = data1.set_index("V")
        data1 = data1.unstack().reset_index()
        data1 = data1.rename(columns={"level_0": "diode", 0: "I-pred"})
        fig, ax = plt.subplots()
        ax = sns.lineplot(
            data=data1, x="V", y="I-pred", hue="diode", palette="tab20c", ax=ax
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_ylim([1e-10, 1])
        ax.set(yscale="log")
        fig.tight_layout()
        col.pyplot(fig)

    I_pred = I_pred_all.sum(axis=0)
    I_true = data_orig["I"].values

    data = data_orig.copy()
    data = data.reset_index(drop=True)
    data.loc[:, "I-pred"] = I_pred

    try:
        r2_score = 100 * metrics.r2_score(np.log10(I_true), np.log10(I_pred))
    except:
        r2_score = np.nan

    errors = 100 * np.abs(I_true - I_pred) / I_pred
    mae_score = np.mean(errors)
    worst_idx = np.argmax(errors)
    max_error = errors[worst_idx]

    st.markdown("### Metrics:")
    st.markdown(
        """
        - R² score: {:.2f}%
        - MAE error: {:.2f}%
        - Max error: {:.2f}%
        - Worst point:
            """.format(
            r2_score, mae_score, max_error,#  data.iloc[worst_idx].to_dict()
        )
    )
    st.write(data.iloc[worst_idx])

    st.markdown("### Optimized parameters:")

    for k, net in enumerate(mixture.nets):
        st.markdown(
            """
        Diode {:d}:

        - Φ: {:s}
        - Peff: {:.4f}
        - n: {:s}
        - Rs: {:s}
        """.format(
                k, str(net.Φ), net.get_peff().item(), str(net.n_net), str(net.rs_net)
            )
        )

    if not to_plot:
        return

    data["T"] = data["T"].astype("int")
    plot(data)

    I_pred_df = pd.DataFrame(I_pred_all.T)
    data = data.reset_index(drop=True)
    temps = np.sort(np.unique(data["T"]))
    for t in temps:
        idxs = data["T"] == t
        data_t = data[idxs]
        I_pred_df_t = I_pred_df[idxs]
        st.markdown(f"### Temp: {t}K")
        col1, col2 = st.columns(2)
        plot(data_t, col1)
        plot_independent(data_t, I_pred_df_t, col2)

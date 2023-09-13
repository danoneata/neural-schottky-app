import streamlit as st


def set_diode_menu(i):
    st.markdown(f"## Diode {i}")
    Φ = st.number_input("Φ", min_value=0.0, value=1.0, step=0.05, key=f"Φ{i}")
    peff = st.number_input("p_eff", min_value=0.0, value=1.0, step=0.05, key=f"peff{i}")
    Rs = st.number_input("Rs", min_value=0.0, value=1.0, step=1.0, key=f"Rs{i}")
    return {
        "Φ": Φ,
        "peff": peff,
        "Rs": Rs,
    }


def main():
    with st.sidebar:
        st.markdown("# Dataset")
        dataset_name = st.selectbox("Select dataset", ["dataset1", "dataset2", "dataset3"])
        temperatures = st.multiselect("Select temperatures", [25, 50, 75, 100], [25, 50, 75, 100])
        st.markdown("---")

        st.markdown("# Parameters")
        num_diodes = st.number_input("Number of diodes", min_value=1, value=1, step=1)

        params = {
            i: set_diode_menu(i)
            for i in range(1, num_diodes + 1)
        }

        # st.markdown("---")
        # st.markdown("# Fitting")


if __name__ == "__main__":
    main()
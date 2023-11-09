from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd



DATA_DIR = Path("data")
TEMPS_DEFAULT = list(range(60, 520, 40))


@dataclass
class Parameters:
    Φ: float
    peff: float
    rs: List[float]
    n: float = 1.03



@dataclass
class DatasetCSV:
    name: str
    diameter: float
    params_init: Parameters
    temps_default: List[int] = field(default_factory=lambda: TEMPS_DEFAULT)
    An: float = 146.0

    @property
    def As(self):
        D = self.diameter * ureg.micrometers
        return compute_As(D.to("cm").magnitude)

    def get_available_temps(self):
        folder = DATA_DIR / self.name
        files = folder.glob("*.csv")
        temps = [int(f.name.split("K")[0]) for f in files]
        return sorted(temps)

    def load(self, temps, min_I=1e-9, max_V=2.5):
        def load_csv_1(T, delimiter=",", decimal="."):
            path = DATA_DIR / self.name / f"{T}K.csv"
            df1 = pd.read_csv(path, delimiter=delimiter, decimal=decimal)
            df1 = df1.rename(columns=lambda s: s.strip())
            df1 = df1[["VOLT1", "CURR1"]].rename(columns={"VOLT1": "V", "CURR1": "I"})
            df1["T"] = T - 273.15  # to Celsius
            return df1

        dfs = [load_csv_1(T) for T in temps]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["I"] >= min_I]
        df = df[df["V"] <= max_V]
        return df


DATASETS = {
    "2023-09-11/pt-600": DatasetCSV(
        name="2023-09-11-pt-600",
        diameter=400.0,
        params_init=Parameters(
            Φ=1.3,
            peff=0.29,
            rs=[50.0],
        ),
    ),
    "2023-09-12/cr-600": DatasetCSV(
        name="2023-09-12-cr-600",
        diameter=400.0,
        params_init=Parameters(
            Φ=1.3,
            peff=0.29,
            rs=[50.0],
        ),
    ),
}

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from skopt.space import Space
from skopt.acquisition import gaussian_ei
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from scipy.stats import norm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utils.helpers import find_closest_points
from LHS.latin_hypercube_sampling import latin_hypercube_sampling
from BO.bayesian_optimisation import bayesian_optimisation

def load_and_transform(filepath, n_components=2, N_only=True):
    """
    Load ligand data from an Excel file and complete PCA, scaling to between 0 and 1.
        filepath (str) :
            Path to the Excel file containing ligand descriptors.
        n_components (int) :
            The number of PCs (dimensionality)
        N_only (bool) :
            If True, restrict the dataset to ligands with "N/N", "N/O", "O/O"-donor atoms.

    Returns:
        df (pandas.DataFrame) :
            The cleaned original dataset after filtering unwanted ligand entries.
        pca_df (pandas.DataFrame) :
            A DataFrame containing the scaled PCA coordinates with columns:
                - "PC1", "PC2", ..., depending on n_components
                - "No." : LKB-bid ligand names carried over from the input file (translateable to SMILES via the Github Excel file)
    """
    df = pd.read_excel(filepath)

    # Exclude all but N donors
    if N_only:
        df = df[df["D1/D2"].isin(["N/N", "N/O", "O/O"])]

    # Remove charged ligands and other NaN values
    df = df[~df["No."].str.contains("ch|SVJ89|SVC", na=False)]
    features = df.select_dtypes(include=[np.number])

    # Fit, transform and scale PCA space
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(features)
    pca_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_components)])
    scaler = MinMaxScaler()
    pca_df[:] = scaler.fit_transform(pca_df)

    # Ensure the Excel file is correctly formatted
    if "No." in df.columns:
        pca_df["No."] = df["No."].values
    else:
        raise KeyError("'No.' column not found in the Excel file.")
    return df, pca_df

def main():
    # Firstly, get the ligand data file and load onto a dataframe with PCA and scaling for latter steps.
    while True:
        filepath = input("Enter the path to the ligand knowledge base file: ").strip()
        if os.path.isfile(filepath):
            break
        print("No existing file with that name.\n")
    df, pca_df = load_and_transform(filepath)
    pc_cols = []
    for col in list(pca_df.columns):
        if col.startswith("PC"):
            pc_cols.append(col)

    # Ask the user whether they require an initial LHS set or if to jump straight to BO.
    print("Enter (1) to continue to LHS for initial space filling suggestions,"
          "or (2) to skip to Bayesian optimisation.")
    while True:
        choice = input("(1) or (2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid input\n")

    # If the user wants to sample with LHS proceed.
    if choice == "1":
        latin_hypercube_sampling(pca_df, pc_cols)

    # Gather in lab yield data from tested ligands, including their standard deviations
    print("\nEnter ligands and yields, and standard deviations:")
    yield_data = pd.DataFrame(columns=["No.", "yield", "std"])

    while True:
        inp = input("Format: <ligand> <yield> <std> OR multiple separated by ';' OR DONE: ").strip()
        if inp.upper() == "DONE":
            break

        # Ensure that any ligands are actually in the data base.
        # Record ligands and yields into a dataframe structure.
        entries = [e.strip() for e in inp.split(";") if e.strip()]
        for entry in entries:
            try:
                ligand, y, std = entry.split()
                y = float(y)
                std = float(std)
                if ligand not in pca_df["No."].values:
                    print(f"Ligand '{ligand}' not found.")
                    continue
                yield_data.loc[len(yield_data)] = [ligand, y, std]
                print(f"Recorded: {ligand} → {y} ({std})")
            except:
                print(f"Invalid entry: {entry}")

    # Run BO
    suggested_samples = bayesian_optimisation(pca_df, yield_data, pc_cols)

    # Ask the user if they would like to see neighbouring ligands (in case ligands are difficult to aquire)
    print("\nOptionally print the closest X ligands to each suggestion.")
    while True:
        try:
            n = int(input("How many neighbours to print: "))
            if n >= 0:
                break
            print("Invalid input\n")
        except:
            print("Invalid input\n")

    print("\nSuggested samples:")

    for s in suggested_samples:
        print(find_closest_points(pca_df, s["coords"], pc_cols, n=n))

if __name__ == "__main__":
    main()
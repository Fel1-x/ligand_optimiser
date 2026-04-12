import numpy as np
import matplotlib.pyplot as plt
# A library of shared functions used across all 3 other python files

def find_closest_points(pca_df, target_coord, pc_cols, n=3):
    """
    Return the n closest points to a target coordinate.
        pca_df (pandas.DataFrame) :
            DataFrame that contains ligand data in principal components
        target_coord (np.array) :
            A 'coordinate' (PC1, PC2, ...) that lies on the pca space 0-1 in each dimension.
        pc_cols (list of str) :
            Column names of the PCA components
        n (int) :
            Number of neighbouring ligands to return

    Returns:
        pandas.DataFrame : contains the rows from pca_df of the
         closest neighbouring ligands to the target coordinate.
    """
    X_all = pca_df[pc_cols].values

    # Compute the distances between each point and the target point
    distances = np.linalg.norm(X_all - target_coord, axis=1)
    closest_indices = np.argsort(distances)[:n]

    # Return the closest n rows in pca_df
    return pca_df.iloc[closest_indices]

def visualise_chemspace(pca_df, pc_cols, ligand_list, backing=None, selection=None):
    """
    A function that prints a visualisation of ligand positions in PCA chemical space.
        pca_df (pandas.DataFrame) :
            DataFrame that contains ligand data in principal components
        pc_cols (list of str) :
            Column names of the PCA components
        ligand_list (list of pandas.DataFrame) :
            A list of 1‑row DataFrames, each representing a single ligand with columns "No.", "PC1", and "PC2".
        backing (np.ndarray or None) :
            Optional 2D array to display as a background heatmap over the PCA space using the imshow() method.
        selection (pandas.DataFrame or None) :
            Optional 1‑row DataFrame representing a selected ligand to highlight in red.

    Returns:
        None : Displays a matplotlib figure showing the PCA chemical space
    """
    fig, ax = plt.subplots()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Ligand Chemical Space (/ %)")

    if backing is not None:
        im = ax.imshow(backing, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    all_points = pca_df[pc_cols].values

    # Overlay full PCA data set
    ax.scatter(all_points[:, 0], all_points[:, 1], c='grey', alpha=0.4, s=20, edgecolor='none')

    for lig in ligand_list:
        ax.scatter(float(lig["PC1"].iloc[0]), float(lig["PC2"].iloc[0]), c='white', edgecolor='black', s=60)

    # Add labels to each plot
        ax.text(float(lig["PC1"].iloc[0]) + 0.01, float(lig["PC2"].iloc[0]) + 0.01, lig["No."].values[0], fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # If printing the AF, print the selected ligand to go along with
    if selection is not None:
        ax.scatter(float(selection["PC1"].iloc[0]), float(selection["PC2"].iloc[0]), c='red', edgecolor='red', s=60)
        ax.text(float(selection["PC1"].iloc[0]) + 0.01, float(selection["PC2"].iloc[0]) + 0.01, selection["No."].values[0], fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.show()
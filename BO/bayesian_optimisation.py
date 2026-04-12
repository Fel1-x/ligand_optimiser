import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utils.helpers import visualise_chemspace
from utils.helpers import find_closest_points
import pandas as pd

def ucb(model, X_candidates, kappa=2.0):
    """
    Upper Confidence Bound (UCB) acquisition function.
    (UCB(x) = mu(x) + kappa * sigma(x))
        model :
            The selected surrogate model.
        X_candidates (np.ndarray) :
            An array of candidate coordinates (ligands in the LKB-bid)
        kappa :
            exploration parameter, baked into ucb function.

    Returns :
        np.ndarray : The ucb aquisition function values at each candidate point.

    """
    mu, sigma = model_mean_std(model, X_candidates)
    return mu + kappa * sigma

def model_mean_std(model, X):
    """
    A function that computes the mean and standard deviation of the surrogate model,
    changing depending on its tree based (RF) or a Gaussian process.
        model :
            The selected surrogate model.
        X (np.ndarray) :
            An array of input coordinates that we want to predict.
    Returns :
        mu (np.ndarray) :
            predicted mean for each X
        sigma (np.ndarray) :
            predicted standard deviation for each X
    """
    # Firstly, try get mean and std directly (if a gp this works well)
    try:
        mu, sigma = model.predict(X, return_std=True)
        return mu, sigma
    except:
        # If it's a tree based model, this will work instead
        all_preds = np.array([tree.predict(X) for tree in model.estimators_])
        mu = np.mean(all_preds, axis=0)
        sigma = np.std(all_preds, axis=0)
        return mu, sigma

def bayesian_optimisation(pca_df, yield_data, pc_cols):
    """
    Perform batch Bayesian optimisation with a selected SM and using the Kriging Believer strategy to complete batch selections.
    Also calls visualisation method to see AF and SM sequentially in 2PCs.
        pca_df (pandas.DataFrame) :
            DataFrame that contains ligand data in principal components
        yield_data (pandas.DataFrame) :
            Experimentally sampled ligands with yields (and std).
        pc_cols (list of str) :
            Column names of the PCA components

    Returns:
        list of dict : Batch of selected ligand suggestions, each containing: {"No.", "coords", "predicted_yield"}.
    """
    batch_size = int(input("How many ligands to suggest in this batch (Kriging Believer): "))

    # Merge the yield data with the pca_df
    combined = yield_data.merge(pca_df, on="No.", how="left")
    X_sampled = combined[pc_cols].values
    y_sampled = combined["yield"].values

    # Create a pool of unsampled ligands by excluding the sampled ligands
    sampled_set = set(yield_data["No."])
    unsampled_df = pca_df[~pca_df["No."].isin(sampled_set)].copy()

    # Seperate the unsampled into PC values and labels
    X_pool = unsampled_df[pc_cols].values
    pool_labels = unsampled_df["No."].values

    selected = [] # This will store chosen ligands during BO

    # Convert the standard deviations given into varience for modelling, accounting for if non is given
    if "std" in combined.columns:
        variance = combined["std"].values ** 2
        for i, v in enumerate(variance):
            if variance[i] <= 0.1:
                variance[i] = 0.1
    else:
        # fallback if std not provided
        variance = np.ones_like(y_sampled) * 1.0

    # To visualise these steps as heat maps, they must be recorded in objects as follows:
    SM_maps = []
    AF_maps = []
    x = np.linspace(0, 1, 160)
    y = np.linspace(0, 1, 160)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Transform the grids as so, so they can be used in the imshow() function
    grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    for i in range(batch_size):
        # Create a model (this can be adjusted to other SM)
        SM = RandomForestRegressor(max_depth=None, min_samples_leaf=2, n_estimators=600, random_state=42, n_jobs=-1)

        # Fit the model, including the varience as "precision"
        SM.fit(X_sampled, y_sampled, sample_weight=1/variance)

        # Select the AF, in this case, upper confidence bound.
        AF = ucb(SM, X_pool, kappa=50)

        # Pick the maximum in the aquision function
        AF_max = np.argmax(AF)
        next_x = X_pool[AF_max]
        next_label = pool_labels[AF_max]

        # Record the predicted value/ std of the selection
        mu_pool, sigma_pool = model_mean_std(SM, X_pool)
        predicted_y = mu_pool[AF_max]
        new_var = sigma_pool[AF_max]**2
        if new_var <= 0.1:
            new_var = 0.1
        variance = np.append(variance, new_var)

        # Store this whole ligand suggestion for printing at the end
        selected.append({"No.": next_label,"coords": next_x,"predicted_yield": predicted_y})

        # Calculate the values across the PC space during this step, and save them for visualisation
        mu_grid, sigma_grid = model_mean_std(SM, grid)
        SM_maps.append(mu_grid.reshape(160, 160))
        af_grid = ucb(SM, grid, kappa=50)
        AF_maps.append(af_grid.reshape(160, 160))

        # Update the data set of sampled ligands
        X_sampled = np.vstack([X_sampled, next_x])
        y_sampled = np.append(y_sampled, predicted_y)

        # Remove the selected ligand from pool of possible ligands
        X_pool = np.delete(X_pool, AF_max, axis=0)
        pool_labels = np.delete(pool_labels, AF_max, axis=0)

    print("SM plots, followed by AF plots and selections will be plotted sequentially")
    null = input("Press Enter to continue to plotting.")

    sams=[]
    for i in range(batch_size):
        for sam in X_sampled[:len(yield_data) + i + 1]:
            sams.append(find_closest_points(pca_df, sam, pc_cols, n=1))

        # Visualise the current model of the chemical space.
        visualise_chemspace(pca_df, pc_cols, sams[:-1], backing=SM_maps[i])

        # Visualise the AF of the chemical space, with the selection highlighted in RED.
        visualise_chemspace(pca_df, pc_cols, sams[:-1], backing=SM_maps[i], selection=sams[-1])

    return selected











import numpy as np
from utils.helpers import find_closest_points
from skopt.sampler import Lhs
from utils.helpers import visualise_chemspace
from skopt.space import Space

def latin_hypercube_sampling(pca_df, pc_cols, dim=2):
    """
    Generate an initial ligand screening design using Latin Hypercube Sampling (LHS),
    optionally incorporating user‑specified ligands.
        pca_df (pandas.DataFrame) :
            DataFrame that contains ligand data in principal components
        pc_cols (list of str) :
            Column names of the PCA components
        dim (int) :
            Dimensionality of the PCA space.

    Returns:
        None : Prints the final design coordinates, the associated nearest ligands,
        both in text and graphically.
    """
    while True:
        try:
            # Assess the number of ligand suggetions that are required
            total_points = int(input("Enter the number of initial ligand suggestions required: "))
            if total_points > 0:
                break
            print("Please enter a positive integer.")
        except:
            print("Invalid input. Enter an integer.")

    # Allow the user to enter any ligands that they would like in the initial screen, LHS will be based around these.
    print("Enter ligands to include in the initial screen (matching input data set), or DONE.\n")

    # Register each ligand, ensuring they appear in the data set provided.
    user_points = []
    while True:
        inp = input("Ligand (or DONE): ").strip()
        if inp.upper() == "DONE":
            break

        row = pca_df[pca_df["No."] == inp]
        if row.empty:
            print("Invalid ligand")
            continue

        user_points.append(row[pc_cols].iloc[0])

    user_points = np.array(user_points)

    # Depending on the number of required suggestions and mandatory ligands:
    # either select the best of the inputted ligands,
    # or suggest new ligands, based on LHS around inputted ligands.
    user_points = np.asarray(user_points).reshape(-1, dim)
    if len(user_points) == 0:
        space = Space([(0.0, 1.0)] * dim)
        sampler = Lhs(criterion="maximin", iterations=1000)
        all_points = np.array(sampler.generate(space.dimensions, total_points))
    elif len(user_points) < total_points:
        lhs = conditional_lhs(user_points, total=total_points)
        all_points = np.vstack([user_points, lhs])
    elif len(user_points) > total_points:
        all_points = select_best(user_points, total_points)
    else:
        all_points = user_points

    print(f"\nFinal {total_points}-point design:")

    pca_df_lhs = pca_df.copy()
    # Output the suggested space filling coordinate and its associated neighbouring ligand(s)
    suggestions = []
    for i, p in enumerate(all_points):
        print(f"{p[0]:.4f}  {p[1]:.4f}")
        suggestions.append(find_closest_points(pca_df_lhs, p, pc_cols, n=1))
        # Remove from the pool, so repetitions arent possible
        pca_df_lhs = pca_df_lhs[pca_df_lhs["No."] != suggestions[i]["No."].iloc[0]]
        print(suggestions[i])

    if dim==2:
        print(suggestions)
        visualise_chemspace(pca_df, pc_cols, suggestions)

def conditional_lhs(user_points, total, dim=2):
    """
    Generate Latin Hypercube Sampling (LHS) points excluding a disk of area around previously selected points
        user_points (np.ndarray) :
            Already‑selected points on ligand space as coords.
        total (int) :
            User selected total number of points (existing + new).
        dim (int) :
            Dimensionality of the pca space.

    Returns:
        np.ndarray :
            Generated LHS points chosen to maximize spacing-filling from existing points.
    """

    R = 0.1
    max_tries = 500

    # Reshape arrays for future use
    user_points = np.asarray(user_points).reshape(-1, dim)

    # Work out how many additional ligands must be selected
    needed = total - len(user_points)

    lhs = Lhs(criterion="maximin", iterations=1000)

    for i in range(max_tries):

        # Generate the required lhs points
        lhs_points = np.array(lhs.generate([(0, 1)] * dim, needed))

        # Check that non of those points are in exclusion zones
        d = np.linalg.norm(lhs_points[:, None, :] - user_points[None, :, :], axis=2)
        valid = np.all(d > R, axis=1)

        print(f"try {i}")

        # If ALL points are valid, return them, otherwise try again till try count is increased
        if np.all(valid):
            return np.vstack([user_points, lhs_points])

    print("Too many tries to generate LHS points, try reducing the required ligand amount.")

def select_best(points, total):
    """
    Select a space‑filling subset of points using maximin sampling,
    this is employed if too many starting ligands are selected
        points (np.ndarray) :
            Candidate coordinates in PCA or latent space.
        total (int) :
            Number of points in the final space‑filling subset.

    Returns:
        np.ndarray :
            Points chosen to maximise minimum pairwise distances.
    """
    points = np.asarray(points)
    n = len(points)

    # Compute a centroid point and start with the point farthest from the centroid
    # in line with literature guidelines for greedy maximin sampling, as used on the LHS sampling fucntion
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    selected = [np.argmax(dists)]

    # Sample the points to maximise the coverage of the PCA space
    while len(selected) < total:
        remaining = np.setdiff1d(np.arange(n), selected)
        # distance to nearest selected point
        for i in remaining:
            min_d.append(np.min(np.linalg.norm(points[i] - points[selected], axis=1)))
        min_d = np.array(min_d)

        selected.append(remaining[np.argmax(min_d)])

    return points[selected]

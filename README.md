# **Ligand Selection Program**

This repository contains a command‑line tool for **ligand selection**.

The program is designed for iterative ligand screening workflows, where experimental results are fed back into the model to guide the next round of ligand suggestions.

## **Basic Usage**

The program is called via:

```
python ligand_selection_program.py
```

and operates as follows.

1. **Load ligand knowledge base**
   - User provides the path to an Excel file containing ligand descriptor data.
   - PCA and scaling are performed.

2. **Choose initial sampling strategy**
   - Enter **(1)** to run Latin Hypercube Sampling.
   - Enter **(2)** to skip directly to Bayesian Optimisation.

3. **Enter experimental data**
   - Input ligands, yields, and standard deviations.
   - Multiple entries can be provided at once using `;` separators.
     - In excel this automation is possible via the following command, where column A contains the name, B the yield and C the standard deviation values:

```
=TEXTJOIN("; ", TRUE, A3:A27 & " " & B3:B27 & " " & C3:C27)
```

4. **Run Bayesian Optimisation**
   - The program suggests a batch of ligands.
   - Surrogate model and acquisition function plots are shown sequentially for each Kriging Believer step.

5. **Optional: print nearest neighbours**
   - The user can specify how many neighbours to list for each suggestion.
   - This allows the substitution of hard to aquire ligands.

---

## **Example Input Format**

When prompted for experimental data:

```
cc01 45.2 1.3
cc02 38.0 2.1; cc05 51.0 1.0
DONE
```

---

## **Outputs**

- A list of ligand suggestions with predicted yields.
- Visualisations of:
  - PCA chemical space with LHS ligands
  - Surrogate model predictions
  - Acquisition function values with selected ligands highlighted in red
- Optional nearest‑neighbour tables for each suggestion.

---

## **Dependencies**

This program uses:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scikit-optimize`

---

## **Project Structure**

```
.
├── ligand_selection_program.py
├── LKB-bid_to_SMILES.xlsx
├── README.md
├── .gitignore
├── BO
│   ├── __init__.py
│   └── bayesian_optimisation.py
├── LHS
│   ├── __init__.py
│   └── latin_hypercube_sampling.py
└── utils
    ├── __init__.py
    └── helpers.py
```

---

## **Notes**

- Variance is used as inverse precision when fitting the surrogate model.
- LKB-bid_to_SMILES.xlsx provides a conversion from LKB-bid names to SMILES chemical strings.

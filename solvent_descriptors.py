

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import sqlite3
from rdkit.Chem import inchi


# Define solvent list (name: SMILES)

solvent_smiles = {
    "Ethanol": "CCO",
    "1-Propanol": "CCCO",
    "2-Propanol": "CC(C)O",
    "4-Methyl-2-Pentanone": "CC(C)CC(=O)C",
    "Acetone": "CC(=O)C",
    "DCM": "ClCCl",
    "THF": "C1CCOC1",
    "Ethyl acetate": "CCOC(=O)C",
    "Toluene": "Cc1ccccc1",
    "n-Hexane": "CCCCCC"
}

# Get all RDKit descriptor names

descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Compute descriptors for each solvent

records = []

for name, smi in solvent_smiles.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"⚠️ Could not parse SMILES for {name}: {smi}")
        continue

    # Calculate descriptors
    values = calculator.CalcDescriptors(mol)
    record = dict(zip(descriptor_names, values))

    # Add metadata
    record["Solvent"] = name
    record["SMILES"] = Chem.MolToSmiles(mol, canonical=True)
    try:
        record["InChI"] = inchi.MolToInchi(mol)
        record["InChIKey"] = inchi.InchiToInchiKey(record["InChI"])
    except:
        record["InChI"] = None
        record["InChIKey"] = None

    records.append(record)

# Create DataFrame
solvent_df = pd.DataFrame(records)

# Reorder columns
meta_cols = ["Solvent", "SMILES", "InChI", "InChIKey"]
cols = meta_cols + [c for c in solvent_df.columns if c not in meta_cols]
solvent_df = solvent_df[cols]

Save to files (CSV, Excel, SQLite)

csv_path = "solvent_descriptors_full.csv"
xlsx_path = "solvent_descriptors_full.xlsx"
sqlite_path = "solvent_descriptors_full.sqlite"

# CSV
solvent_df.to_csv(csv_path, index=False)
# Excel
solvent_df.to_excel(xlsx_path, index=False)
# SQLite
conn = sqlite3.connect(sqlite_path)
solvent_df.to_sql("solvent_descriptors", conn, if_exists="replace", index=False)
conn.close()

# summary
print("✅ Solvent descriptor database generated successfully!")
print(f"Total solvents: {len(solvent_df)}")
print(f"Total descriptors per solvent: {len(descriptor_names)}")
print("\nFiles created:")
print(f"- {csv_path}")
print(f"- {xlsx_path}")
print(f"- {sqlite_path}")



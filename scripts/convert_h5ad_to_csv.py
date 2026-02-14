#!/usr/bin/env python3
"""
Convert H5AD files to CSV format
"""

import pandas as pd
import anndata
import os
import sys

def convert_h5ad_to_csv(h5ad_path, output_dir="."):
    """
    Convert an H5AD file to CSV format, extracting:
    - Expression matrix (X) as CSV
    - Observations metadata as CSV
    - Variables metadata as CSV
    """
    print(f"Loading {h5ad_path}...")

    # Load the H5AD file
    adata = anndata.read_h5ad(h5ad_path)

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(h5ad_path))[0]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert expression matrix to CSV
    if hasattr(adata, 'X'):
        print(f"Converting expression matrix (shape: {adata.shape})...")
        expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                              index=adata.obs_names,
                              columns=adata.var_names)
        expr_output = os.path.join(output_dir, f"{base_name}_expression.csv")
        expr_df.to_csv(expr_output)
        print(f"Saved expression matrix to: {expr_output}")

    # Convert observations metadata to CSV
    if hasattr(adata, 'obs') and len(adata.obs) > 0:
        print(f"Converting observations metadata ({len(adata.obs)} observations)...")
        obs_output = os.path.join(output_dir, f"{base_name}_observations.csv")
        adata.obs.to_csv(obs_output)
        print(f"Saved observations metadata to: {obs_output}")

    # Convert variables metadata to CSV
    if hasattr(adata, 'var') and len(adata.var) > 0:
        print(f"Converting variables metadata ({len(adata.var)} variables)...")
        var_output = os.path.join(output_dir, f"{base_name}_variables.csv")
        adata.var.to_csv(var_output)
        print(f"Saved variables metadata to: {var_output}")

    # Convert uns (unstructured) data if it contains useful information
    if hasattr(adata, 'uns') and len(adata.uns) > 0:
        print(f"Converting unstructured metadata ({len(adata.uns)} items)...")
        uns_output = os.path.join(output_dir, f"{base_name}_metadata.json")
        import json
        # Convert to JSON since uns can contain complex nested structures
        with open(uns_output, 'w') as f:
            json.dump({k: str(v) for k, v in adata.uns.items()}, f, indent=2)
        print(f"Saved unstructured metadata to: {uns_output}")

    print(f"Successfully converted {h5ad_path} to CSV format!")
    return True

def main():
    # Define the H5AD files to convert
    h5ad_files = [
        r"E:\CODE\trishift\src\data\norman\perturb_processed.h5ad"
    ]

    output_directory = "./csv_output"

    # Convert each file
    for h5ad_file in h5ad_files:
        if os.path.exists(h5ad_file):
            try:
                convert_h5ad_to_csv(h5ad_file, output_directory)
                print("-" * 50)
            except Exception as e:
                print(f"Error converting {h5ad_file}: {str(e)}")
                print("-" * 50)
        else:
            print(f"File not found: {h5ad_file}")
            print("-" * 50)

if __name__ == "__main__":
    main()
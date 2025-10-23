#!/usr/bin/env python3
"""
Script to fix parquet metadata by removing problematic _type List definitions
that cause TypeError in HuggingFace datasets library.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import json
import os
from pathlib import Path

def fix_parquet_metadata(file_path):
    """Fix parquet metadata by removing _type List definitions."""
    print(f"Processing: {file_path}")
    
    # Read the parquet file
    table = pq.read_table(file_path)
    schema = table.schema
    
    # Get the metadata
    metadata = dict(schema.metadata) if schema.metadata else {}
    
    # Fix the huggingface metadata
    if b'huggingface' in metadata:
        try:
            hf_metadata = json.loads(metadata[b'huggingface'].decode('utf-8'))
            
            # Convert List to Sequence for compatibility
            if 'info' in hf_metadata and 'features' in hf_metadata['info']:
                features = hf_metadata['info']['features']
                for feature_name, feature_def in features.items():
                    if isinstance(feature_def, dict) and feature_def.get('_type') == 'List':
                        # Convert List to Sequence for compatibility
                        feature_def['_type'] = 'Sequence'
                        print(f"  Converted {feature_name} from List to Sequence")
            
            # Update the metadata
            metadata[b'huggingface'] = json.dumps(hf_metadata).encode('utf-8')
            
        except Exception as e:
            print(f"  Error processing metadata: {e}")
            return False
    
    # Create new schema with fixed metadata
    new_schema = pa.schema(schema, metadata=metadata)
    
    # Write the fixed parquet file
    table_with_fixed_schema = table.cast(new_schema)
    pq.write_table(table_with_fixed_schema, file_path)
    
    print(f"  Fixed metadata for: {file_path}")
    return True

def main():
    """Fix all parquet files in the dataset directory."""
    dataset_dir = Path("/home/shadeform/.cache/huggingface/lerobot/asgard-robot/s0101-test-v2.1")
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Find all parquet files
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found")
        return
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Process each file
    success_count = 0
    for file_path in parquet_files:
        try:
            if fix_parquet_metadata(file_path):
                success_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nSuccessfully processed {success_count}/{len(parquet_files)} files")

if __name__ == "__main__":
    main()

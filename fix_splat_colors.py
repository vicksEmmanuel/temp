import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os

def fix_ply_colors(input_path, output_path):
    print(f"Reading {input_path}...")
    plydata = PlyData.read(input_path)
    
    # Extract vertex elements
    vertex = plydata.elements[0]
    data = vertex.data.copy()
    
    # Identify f_dc properties
    f0 = np.asarray(data['f_dc_0'])
    f1 = np.asarray(data['f_dc_1'])
    f2 = np.asarray(data['f_dc_2'])
    
    # Check if they look like logit values (not clamped to [0,1])
    if f0.min() < -1 or f0.max() > 2:
        print("Detected logit-space colors. Applying sigmoid(f_dc_0..2)...")
        
        # Apply sigmoid: 1 / (1 + exp(-x))
        data['f_dc_0'] = 1.0 / (1.0 + np.exp(-f0))
        data['f_dc_1'] = 1.0 / (1.0 + np.exp(-f1))
        data['f_dc_2'] = 1.0 / (1.0 + np.exp(-f2))
        
        print(f"Saving fixed splat to {output_path}...")
        el = PlyElement.describe(data, 'vertex')
        PlyData([el]).write(output_path)
        print("Done! This splat should now look bright in standard viewers.")
    else:
        print("Splat colors already appear to be in [0,1] range. No changes made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix 4D Gaussian Splat colors for viewers")
    parser.add_argument("input", help="Path to input .ply file")
    parser.add_argument("--output", "-o", help="Path to output .ply file (default: input_fixed.ply)")
    
    args = parser.parse_args()
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed{ext}"
        
    fix_ply_colors(args.input, args.output)

import json
import sys
import io
import traceback
import numpy as np

notebook_path = r"c:\Users\spada\Downloads\ilya_reading_package\PCA_Multivariate_Gaussian_Analysis.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

global_env = {}

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        print(f"Executing cell {idx}...")
        
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture
        
        try:
            exec(code_str, global_env)
            output_text = stdout_capture.getvalue()
            
            cell["execution_count"] = idx // 2
            cell["outputs"] = []
            
            if output_text:
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True)
                })
        except Exception as e:
            output_text = stdout_capture.getvalue()
            err_msg = traceback.format_exc()
            print(f"Error in cell {idx}: {err_msg}", file=sys.stderr)
            cell["outputs"] = [{
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": err_msg.splitlines()
            }]
        finally:
            sys.stdout = sys.__stdout__

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully executed and updated notebook outputs: {notebook_path}")

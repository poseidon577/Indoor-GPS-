import os
import sys
import importlib
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.join(base_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

modules_to_check = {
    "Environment": "VLCL_AI.environment",
    "Physics": "VLCL_AI.physics",
    "Communication": "VLCL_AI.communication",
    "Localization": "VLCL_AI.localization",
    "Spectrum Management": "VLCL_AI.communication.resource_allocation",
    "Adaptive Modulation": "VLCL_AI.adaptive",
    "Power Optimization": "VLCL_AI.integrated_vlcl",
    "Joint Optimization": "VLCL_AI.integrated_vlcl",
    "Validation": "VLCL_AI.reproduction"
}

def check_module(module_name, import_path):
    try:
        mod = importlib.import_module(import_path)
        # Check if it has actual python files
        module_dir = os.path.dirname(mod.__file__)
        py_files = [f for f in os.listdir(module_dir) if f.endswith('.py') and f != '__init__.py']
        if len(py_files) > 0:
            return "Implemented"
        else:
            return "Placeholder"
    except ImportError as e:
        return f"Unused"
    except Exception as e:
        return f"Partially implemented"

def main():
    report = {}
    for mod_name, imp_path in modules_to_check.items():
        status = check_module(mod_name, imp_path)
        report[mod_name] = status
        
    try:
        import VLCL_AI.adaptive.modulation
        report["Adaptive Modulation"] = "Implemented"
    except:
        report["Adaptive Modulation"] = "Placeholder"
        
    try:
        import VLCL_AI.integrated_vlcl.optimizer
        report["Joint Optimization"] = "Implemented"
        report["Power Optimization"] = "Implemented"
    except:
        report["Joint Optimization"] = "Placeholder"
        report["Power Optimization"] = "Placeholder"

    print("Inspection Results:")
    for k, v in report.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(results_dir, "step01_inspection.json"), "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()

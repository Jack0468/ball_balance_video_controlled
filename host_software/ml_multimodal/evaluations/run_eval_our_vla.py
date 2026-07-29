import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_eval_baseline_vla import VLAEvaluator

if __name__ == "__main__":
    # Our VLA uses the exact same evaluation sequence, but will load our RL-finetuned weights
    # and save to labels_sequential_our_vla.csv
    evaluator = VLAEvaluator(is_baseline=False)
    
    # Normally we would do:
    # evaluator.vla_model.load_state_dict(torch.load("models/vla_v1/rl_finetuned_vla.pth"))
    
    evaluator.run_sequence()

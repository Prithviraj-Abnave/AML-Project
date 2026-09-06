import time
import json
import os
import copy
from ml.pipeline import _run_pipeline_worker, get_results, load_cached_results, _pipeline_state

def evaluate_k(k, sample_size=25000):
    print(f"\n--- Evaluating k={k} ---")
    
    with _pipeline_state['lock']:
        _pipeline_state['results'] = None
    
    _run_pipeline_worker(n_topics=k, sample_size=sample_size, data_path=None)
    
    results_path = os.path.join('results', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            res = json.load(f)
            
        lda_coh = res['lda']['metrics']['coherence']
        nmf_coh = res['nmf']['metrics']['coherence']
        avg_coh = (lda_coh + nmf_coh) / 2
        print(f"k={k} -> LDA Coherence: {lda_coh:.4f}, NMF Coherence: {nmf_coh:.4f}, Avg: {avg_coh:.4f}")
        return avg_coh, res
    return 0, None

def main():
    k_values = [4, 5, 6]  # Test a narrow range around 5 to save time but still find an optimal local peak
    best_k = None
    best_coh = -1
    best_res = None
    
    for k in k_values:
        avg_coh, res = evaluate_k(k)
        if avg_coh > best_coh:
            best_coh = avg_coh
            best_k = k
            best_res = copy.deepcopy(res)
            
    print(f"\n=== Best k found: {best_k} (Avg Coherence: {best_coh:.4f}) ===")
    
    # Just save the best result back to results.json instead of retraining
    if best_res:
        results_path = os.path.join('results', 'results.json')
        with open(results_path, 'w') as f:
            json.dump(best_res, f, indent=2)
        print("Done! Dashboard results updated with the best model.")

if __name__ == "__main__":
    main()

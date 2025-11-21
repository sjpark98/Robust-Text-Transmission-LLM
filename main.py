import numpy as np
import pandas as pd
from config import Config
from model import Proposed
from tools import Similarity, calculate_bleu
from solver import GptSolver

def main():
    config = Config()
    config.print_options()
    
    test_data = pd.read_csv("./dataset/sentences.csv", sep=',')
    proposed = Proposed(config, test_data)
    marked_set, random_marked_set = proposed.system_model()
    
    similarity = Similarity()
    solver = GptSolver(api_key=config.param['api_key'])
    
    results = {
        'sim': [], 'sim_rand': [],
        'bleu': [], 'bleu_rand': []
    }

    for i in range(config.param['num_test']):
        print(f"No. {i+1}")
        ori_sentence = test_data.iloc[[i]].to_string(index=False, header=False)
        cleaned_ori = ori_sentence.lower()
        
        prop_restored = solver.restore_sentence(marked_set[i])
        rand_restored = solver.restore_sentence(random_marked_set[i])
        
        print(f"Prop Marked: {marked_set[i]}")
        print(f"Rand Marked: {random_marked_set[i]}")
        print("-" * 80)
        print(f"Original:  {ori_sentence}")
        print(f"Prop LLM:  {prop_restored}")
        print(f"Rand LLM:  {rand_restored}")
        print("-" * 80)

        s = similarity.compute_score(cleaned_ori, prop_restored.lower()).item()
        rand_s = similarity.compute_score(cleaned_ori, rand_restored.lower()).item()
        
        b = calculate_bleu(cleaned_ori, prop_restored.lower())
        rand_b = calculate_bleu(cleaned_ori, rand_restored.lower())

        results['sim'].append(s)
        results['sim_rand'].append(rand_s)
        results['bleu'].append(b)
        results['bleu_rand'].append(rand_b)

        print(f"Prop - Sim: {s:.3f}, BLEU: {b:.3f}")
        print(f"Rand - Sim: {rand_s:.3f}, BLEU: {rand_b:.3f}")
        print("\n")

    print("=" * 80)
    print(f"Prop Avg Similarity: {np.mean(results['sim']):.4f}")
    print(f"Prop Avg BLEU:       {np.mean(results['bleu']):.4f}")
    print("-" * 80)
    print(f"Rand Avg Similarity: {np.mean(results['sim_rand']):.4f}")
    print(f"Rand Avg BLEU:       {np.mean(results['bleu_rand']):.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
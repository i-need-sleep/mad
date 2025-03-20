import argparse

from .compute_mad import compute_mad

def main():
    parser = argparse.ArgumentParser(description='Compute MAD score between two sets of embeddings')
    parser.add_argument('--eval_dir', type=str, help='Directory containing evaluation files')
    parser.add_argument('--ref_dir', type=str, help='Directory containing reference files')
    parser.add_argument('--eval_embs_dir', type=str, help='Directory containing evaluation embeddings')
    parser.add_argument('--ref_embs_dir', type=str, help='Directory containing reference embeddings')
    parser.add_argument('--log_csv', type=str, help='Path to log file')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for featurization')
    parser.add_argument('--model_name', type=str, default='mert_330m', help='Model name')
    parser.add_argument('--layer', type=int, default=24, help='Layer to extract embeddings from')
    parser.add_argument('--aggregation', type=str, default='max', help='Aggregation method for embeddings')
    args = parser.parse_args()

    compute_mad(
        eval_dir=args.eval_dir, 
        ref_dir=args.ref_dir, 
        eval_embs_dir=args.eval_embs_dir, 
        ref_embs_dir=args.ref_embs_dir, 
        log_csv=args.log_csv,
        batch_size=args.batch_size, 
        model_name=args.model_name, 
        layer=args.layer, 
        aggregation=args.aggregation
    )

if __name__ == '__main__':
    main()
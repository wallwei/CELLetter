import os
import json
import logging
import random
import numpy as np
import torch
from data.data_loader import GetLRIDataset
from models.glorimodel import EnhancedDecoderWithResidues
from trainers.trainer import train_and_evaluate, evaluate
from utils.logger import setup_logger
from config.params import *
import time
import psutil

def main():
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    # Setup logger
    logger = setup_logger()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "roc_curves"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "pr_curves"), exist_ok=True)

    # Create summary file
    auc_aupr_file = os.path.join(RESULTS_DIR, "auc_aupr_results.csv")
    with open(auc_aupr_file, 'w') as f:
        f.write("repeat,fold,AUC,AUPR\n")

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Model name: GLORI (Global and Local feature fusion with gating for Receptor-Ligand Interaction prediction)")

    # Initialize dataset
    dataset = GetLRIDataset(DATA_DIR, RESIDUE_DIR)

    # Get all ligand-receptor pair information
    all_pairs = []
    for l, r, label in dataset.lri_data:
        all_pairs.append((l, r, label))

    # Create index mappings for ligands and receptors
    ligands = sorted(set(l for l, r, _ in all_pairs))
    receptors = sorted(set(r for l, r, _ in all_pairs))
    ligand_to_idx = {ligand: idx for idx, ligand in enumerate(ligands)}
    receptor_to_idx = {receptor: idx for idx, receptor in enumerate(receptors)}

    # Save index mappings
    with open(os.path.join(RESULTS_DIR, 'ligand_index_mapping.json'), 'w') as f:
        json.dump(ligand_to_idx, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'receptor_index_mapping.json'), 'w') as f:
        json.dump(receptor_to_idx, f, indent=2)
    logging.info("Ligand and receptor index mappings saved")

    # Multiple repeats of five-fold cross-validation
    all_repeat_metrics = []
    best_auc = 0.0
    best_model_path = None
    best_repeat_fold = None

    for repeat in range(1, N_REPEATS + 1):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Starting {repeat}th repeat of five-fold cross-validation")
        logging.info(f"{'=' * 50}")

        # Set different random seeds
        random_seed = SEED + repeat * 10
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        fold_loaders = dataset.get_kfold_dataloader(batch_size=BATCH_SIZE, n_splits=N_SPLITS)
        repeat_metrics = []

        for fold, (train_loader, test_loader) in enumerate(fold_loaders, 1):
            logging.info(f"\n{'=' * 30}")
            logging.info(f"Repeat {repeat} - Fold {fold}")
            logging.info(f"{'=' * 30}")

            # Initialize model
            model = EnhancedDecoderWithResidues().to(DEVICE)

            # Train and evaluate
            model, fold_auc = train_and_evaluate(model, train_loader, test_loader, DEVICE,
                                                num_epochs=NUM_EPOCHS, fold=f"{repeat}_{fold}")

            metrics, roc_data, pr_data = evaluate(model, test_loader, DEVICE)

            # Save ROC curve data
            roc_file = os.path.join(RESULTS_DIR, "roc_curves", f"repeat{repeat}_fold{fold}_roc.csv")
            roc_data.to_csv(roc_file, index=False)

            # Save PR curve data
            pr_file = os.path.join(RESULTS_DIR, "pr_curves", f"repeat{repeat}_fold{fold}_pr.csv")
            pr_data.to_csv(pr_file, index=False)

            # Save AUC and AUPR to file
            with open(auc_aupr_file, 'a') as f:
                f.write(f"{repeat},{fold},{metrics['auc']:.6f},{metrics['aupr']:.6f}\n")

            # Check if this is the best model
            if fold_auc > best_auc:
                best_auc = fold_auc
                best_model_path = os.path.join(RESULTS_DIR, f'best_model_repeat{repeat}_fold{fold}.pth')
                best_repeat_fold = (repeat, fold)
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model found, AUC: {fold_auc:.4f}, saved at {best_model_path}")

            # Save results
            repeat_metrics.append({
                "acc": metrics['acc'],
                "f1": metrics['f1'],
                "auc": metrics['auc'],
                "aupr": metrics['aupr'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "mcc": metrics['mcc'],
                "loss": metrics.get('loss', 0)
            })
            logging.info(f"Repeat {repeat} - Fold {fold} results: "
                         f"Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | "
                         f"AUC: {metrics['auc']:.4f} | AUPR: {metrics['aupr']:.4f} | "
                         f"Pre: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
                         f"MCC: {metrics['mcc']:.4f}")

        # Save this repeat's results
        repeat_results_file = os.path.join(RESULTS_DIR, f'repeat_{repeat}_results.json')
        with open(repeat_results_file, 'w') as f:
            json.dump({
                "repeat": repeat,
                "folds": [{"fold": i + 1, "metrics": m} for i, m in enumerate(repeat_metrics)]
            }, f, indent=2)

        # Calculate average performance for this repeat
        avg_metrics = {
            'acc': np.mean([m['acc'] for m in repeat_metrics]),
            'f1': np.mean([m['f1'] for m in repeat_metrics]),
            'auc': np.mean([m['auc'] for m in repeat_metrics]),
            'aupr': np.mean([m['aupr'] for m in repeat_metrics]),
            'precision': np.mean([m['precision'] for m in repeat_metrics]),
            'recall': np.mean([m['recall'] for m in repeat_metrics]),
            'mcc': np.mean([m['mcc'] for m in repeat_metrics])
        }

        std_metrics = {
            'acc': np.std([m['acc'] for m in repeat_metrics]),
            'f1': np.std([m['f1'] for m in repeat_metrics]),
            'auc': np.std([m['auc'] for m in repeat_metrics]),
            'aupr': np.std([m['aupr'] for m in repeat_metrics]),
            'precision': np.std([m['precision'] for m in repeat_metrics]),
            'recall': np.std([m['recall'] for m in repeat_metrics]),
            'mcc': np.std([m['mcc'] for m in repeat_metrics])
        }

        logging.info(f"\nRepeat {repeat} average results:")
        logging.info(f"Accuracy: {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
        logging.info(f"F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
        logging.info(f"AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
        logging.info(f"AUPR: {avg_metrics['aupr']:.4f} ± {std_metrics['aupr']:.4f}")
        logging.info(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
        logging.info(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
        logging.info(f"MCC: {avg_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")

    # Calculate average performance across all repeats
    flat_metrics = [m for repeat_metrics in all_repeat_metrics for m in repeat_metrics]
    avg_metrics = {
        'acc': np.mean([m['acc'] for m in flat_metrics]),
        'f1': np.mean([m['f1'] for m in flat_metrics]),
        'auc': np.mean([m['auc'] for m in flat_metrics]),
        'aupr': np.mean([m['aupr'] for m in flat_metrics]),
        'precision': np.mean([m['precision'] for m in flat_metrics]),
        'recall': np.mean([m['recall'] for m in flat_metrics]),
        'mcc': np.mean([m['mcc'] for m in flat_metrics])
    }

    std_metrics = {
        'acc': np.std([m['acc'] for m in flat_metrics]),
        'f1': np.std([m['f1'] for m in flat_metrics]),
        'auc': np.std([m['auc'] for m in flat_metrics]),
        'aupr': np.std([m['aupr'] for m in flat_metrics]),
        'precision': np.std([m['precision'] for m in flat_metrics]),
        'recall': np.std([m['recall'] for m in flat_metrics]),
        'mcc': np.std([m['mcc'] for m in flat_metrics])
    }

    logging.info(f"\n{'=' * 50}")
    logging.info(f"Average results of {N_REPEATS} repeats of five-fold cross-validation:")
    logging.info(f"Accuracy: {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
    logging.info(f"F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    logging.info(f"AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    logging.info(f"AUPR: {avg_metrics['aupr']:.4f} ± {std_metrics['aupr']:.4f}")
    logging.info(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    logging.info(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    logging.info(f"MCC: {avg_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")
    logging.info(f"{'=' * 50}")

    # Save all results
    all_results_file = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump({
            'all_repeats': all_repeat_metrics,
            'average': avg_metrics,
            'std_dev': std_metrics,
            'best_model': {
                'path': best_model_path,
                'repeat_fold': best_repeat_fold,
                'auc': best_auc
            }
        }, f, indent=2)
    logging.info(f"All results saved to {all_results_file}, best model saved at {best_model_path}")

    # Prompt user to run prediction script
    logging.info("\nTraining completed! Please run the prediction script using the best model:")
    logging.info(
        f"python predict.py --model_path {best_model_path} --data_dir {DATA_DIR} --residue_dir {RESIDUE_DIR} --output_dir {RESULTS_DIR}")

    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"Memory used: {memory_used} bytes")

if __name__ == "__main__":
    main()
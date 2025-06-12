import pandas as pd
from sklearn.metrics import f1_score
import argparse

def main(file_path):
    df = pd.read_csv(file_path)

    y_true = df['labels']
    y_pred = df['predict']

    score = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 Score: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Macro F1 Score")
    parser.add_argument('--file', type=str, default='./Result/gem2fl.csv', help='Path to the CSV file containing the predictions and labels')
    args = parser.parse_args()

    main(args.file)

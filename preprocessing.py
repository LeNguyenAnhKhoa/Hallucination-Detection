import pandas as pd
from ftfy import fix_text
import re
import argparse

def clean_claim(text):
    return re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text).strip()

def clean_reference(text):
    return re.sub(r"^\[\s*\d+\s*\]:\s*", "", text).strip()

def main(subtask1_path, subtask2_path, output_path):
    df = pd.read_csv(subtask1_path)
    df = df.drop(columns=['question', 'answer', 'justification'])
    df["claim_clean"] = df["claim"].apply(clean_claim).apply(fix_text)
    df["reference_clean"] = df["reference"].apply(clean_reference).apply(fix_text)
    df['label'] = df['label'].replace({
        'entail': 'Entailment',
        'contra': 'Contradiction',
        'unver': 'Unverifiable'
    })

    df2 = pd.read_csv(subtask2_path)
    df2_renamed = df2.rename(columns={'label': 'labels'})
    df = df.merge(df2_renamed[['claim', 'labels']], on='claim', how='left')
    df = df.drop(columns=['claim', 'reference'])
    df['labels'] = df['labels'].replace({
        'negat': 'Opposite meaning',
        'misinter': 'Misrepresentation',
        'relunvef': 'Related but unverifiable',
        'entail': 'Entailment',
        'entierr': 'Entity error',
        'unrelunvef': 'Unrelated and unverifiable',
        'numerr': 'Numeric error',
        'missinfo': 'Missing information'
    })

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and merge claim/reference datasets.")
    parser.add_argument("--subtask1", type=str, required=True, help="Path to subtask1 CSV file")
    parser.add_argument("--subtask2", type=str, required=True, help="Path to subtask2 CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output cleaned CSV file")
    
    args = parser.parse_args()
    main(args.subtask1, args.subtask2, args.output)

import pandas as pd
from ftfy import fix_text
import re

def clean_claim(text):
    return re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text).strip()

def clean_reference(text):
    return re.sub(r"^\[\s*\d+\s*\]:\s*", "", text).strip()

df = pd.read_csv("./Data/subtask1_train_batch3.csv")
df = df.drop(columns=['question', 'answer', 'justification'])
df["claim_clean"] = df["claim"].apply(clean_claim).apply(fix_text)
df["reference_clean"] = df["reference"].apply(clean_reference).apply(fix_text)
df['label'] = df['label'].replace({
    'entail': 'Entailment',
    'contra': 'Contradiction',
    'unver': 'Unverifiable'
})

df2 = pd.read_csv('./Data/subtask2_train_batch3.csv')
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

df.to_csv("./Data/cleaned_data3.csv", index=False)
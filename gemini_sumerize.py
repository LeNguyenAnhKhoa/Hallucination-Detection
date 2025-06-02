import argparse
from google import genai
from google.genai import types
import pandas as pd
import time

def get_prediction(claim, reference, model_name):
    prompt_sum = """
    You are a summarizer that extracts and condenses the key points from the given text into a clear and concise summary. Your task is to summarize the claim and reference from the following academic article for research purposes. Please present clearly separating 'claim' and 'reference'.
    You will be given two inputs: claim, reference.
    Output format:
    claim:
    reference:
    Input:
    """

    client = genai.Client(api_key="")

    config = types.GenerateContentConfig(
        temperature=0
    )

    full_prompt = f"{prompt_sum} \n claim: {claim} \n reference: {reference}"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main(output_path, sleep_time, model_name):
    df = pd.read_csv('Data/cleaned_data.csv')

    predictions = []

    for idx, row in df.iterrows():
        claim = row['claim_clean']
        reference = row['reference_clean']
        prediction = get_prediction(claim, reference, model_name)
        predictions.append(prediction)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}...")

        time.sleep(sleep_time)

    df['text'] = predictions

    df = df.drop(columns=['claim_clean', 'reference_clean'])

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize academic articles using Gemini.")
    parser.add_argument('--model', type=str, default = 'gemini-2.5-flash-preview-04-17', help='Model name to use (e.g., "models/gemini-1.5-pro-latest")')
    parser.add_argument('--output', type=str, default = './Data/gem25fl_sum.csv', help='Path to output CSV file')
    parser.add_argument('--sleep_time', type=float, default=6.3, help='Sleep time (in seconds) between requests')
    main(parser.parse_args().output, parser.parse_args().sleep_time, parser.parse_args().model)
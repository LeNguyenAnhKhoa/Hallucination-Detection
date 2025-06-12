import argparse
from google import genai
from google.genai import types
import pandas as pd
import time

def get_prediction(text, model_name):
    prompt_prefix = """
        You are an annotator concerned that the claim may not align with the reference. Your task is to determine whether the reference entail, contradict, or is unverifiable to the claim.
        You will be given two inputs: claim, reference.
        You are asked to evaluate the generated text looking at the input text and the target text.
        A reference entails a claim if the information in the reference directly supports the claim being true and there is no conflicting information. The information logically leads to the claim being correct. A reference contradicts a claim if it provides information that directly disproves or disagrees with any part of the claim. This includes stating different entities, numeric values, or relations compared to the claim. A reference is unverifiable to a claim if it does not provide enough information to determine whether the claim is true or false. This happens if the reference is unrelated, missing key details, or too ambiguous.
        Example:
        ##
        claim: 8. X-ray Techniques: X-ray Absorption and Diffraction: Methods like extended X-ray absorption fine structure, X-ray diffraction, and low-angle scattering are used to gain and analyze experimental data for material examination .
        reference: Ways to gain and analyze experimental data obtained by X-ray techniques used in material examination are described. Emphasis is on the methods of extended X-ray absorption fine structure, X-ray diffraction, and X-ray low-angle scattering.
        answer: Entailment
        ##
        claim: ### Machine Condition Monitoring - **Predictive Maintenance**: ML algorithms are not effective in predicting future damages in technical machines, such as turbines and pumps, as they fail to accurately model and extrapolate damage mechanisms based on sensor data.
        reference: Many technical machines are instrumented. Temperatures, pressures, flow rates, vibrations and so on are measured and centrally archived. These data can be used to reliably predict future damages several days in advance. A self-learning mathematical method is used to do this, which models the machine and can extrapolate the damage mechanism into the future. Examples include turbines, pumps and catalytic reactors that will be treated in this paper.
        answer: Contradiction
        ##
        claim: Ensuring that chatbots can handle a wide range of queries and provide accurate, relevant information is crucial for their effectiveness .
        reference:  In this paper we learn how to manage a dialogue relying on discourse of its utterances. We define extended discourse trees, introduce means to manipulate with them, and outline scenarios of multi-document navigation to extend the abilities of the interactive information retrieval-based chat bot. We also provide evaluation results of the comparison between conventional search and chat bot enriched with the multi-document navigation.
        answer: Unverifiable
        ##
        Your response only answer 'Entailment' or 'Contradiction' or 'Unverifiable'
        Input:
    """

    client = genai.Client(api_key="")

    config = types.GenerateContentConfig(
        temperature=0
    )

    full_prompt = f"{prompt_prefix} \n {text} \n answer:"
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

def main(output_path, sleep_time, model_name, path):
    df = pd.read_csv(path)

    predictions = []

    for idx, row in df.iterrows():
        text = row['text']
        prediction = get_prediction(text, model_name)
        predictions.append(prediction)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}...")

        time.sleep(sleep_time)

    df['predict'] = predictions

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference summerized data using Gemini.")
    parser.add_argument('--model', type=str, default = 'gemini-2.5-flash-preview-04-17', help='Model name to use (e.g., "models/gemini-1.5-pro-latest")')
    parser.add_argument('--output', type=str, default = './Result/gem_sum_result.csv', help='Path to output CSV file')
    parser.add_argument('--sleep_time', type=float, default=6.3, help='Sleep time (in seconds) between requests')
    parser.add_argument('--path', type=str, default = '../Data/gem25fl_sum.csv', help='Path to input CSV file')
    main(parser.parse_args().output, parser.parse_args().sleep_time, parser.parse_args().model, parser.parse_args().path)
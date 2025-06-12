import argparse
from google import genai
from google.genai import types
import pandas as pd
import time

def get_prediction(claim, reference, model_name):
    prompt_prefix = """
    You are an annotator concerned that the claim may not align with the reference. Your task is to determine whether the reference entail, contradict, or is unverifiable to the claim.
    You will be given two inputs: claim, reference.
    You are asked to evaluate the generated text looking at the input text and the target text.
    A reference entails a claim if the information in the reference directly supports the claim being true and there is no conflicting information. The information logically leads to the claim being correct. A reference contradicts a claim if it provides information that directly disproves or disagrees with any part of the claim. This includes stating different entities, numeric values, or relations compared to the claim. A reference is unverifiable to a claim if it does not provide enough information to determine whether the claim is true or false. This happens if the reference is unrelated, missing key details, or too ambiguous.
    Example:
    ##
    claim: 8. X-ray Techniques: X-ray Absorption and Diffraction: Methods like extended X-ray absorption fine structure, X-ray diffraction, and low-angle scattering are used to gain and analyze experimental data for material examination .
    reference: Ways to gain and analyze experimental data obtained by X-ray techniques used in material examination are described. Emphasis is on the methods of extended X-ray absorption fine structure, X-ray diffraction, and X-ray low-angle scattering.
    justification: The claim is supported by the reference. - "Ways to gain and analyze experimental data obtained by X-ray techniques used in material examination are described. Emphasis is on the methods of extended X-ray absorption fine structure, X-ray diffraction, and X-ray low-angle scattering."
    answer: Entailment
    ##
    claim: Disadvantages of Dropdown Menus: Dropdown menus can hide information, requiring users to perform additional actions to view all options .
    reference: Hierarchical menus are a common feature of the user interface for interactive software. Dynamic menus allow users to add items to the menu structure. Such dynamic menus are subject to usability problems of hiding information because of overlapping data and of requiring large in-line movement of the mouse or input device. This paper presents an improved design for menu display and interaction to provide easier viewing and navigation.
    justification: The reference clearly states that dynamic (dropdown) menus may hide information and require large in-line movement, which directly supports the claims two parts: that dropdown menus can obscure information and require additional user actions to reveal it.
    answer: Entailment
    ##
    claim: Application: Focuses on long-term rehabilitation and recovery, offering outpatient services close to the patient's home .
    reference: The VHA polytrauma system of care is a comprehensive, integrated treatment program, based on decades of research and clinical experience in geriatric care and in the rehabilitation of individuals with acute and chronic disability. The PSC uses an interdisciplinary team model approach, and an array of outpatient rehabilitation services close to the patient's home is offered at rehabilitation sites within the PSC. Copyright © 2010 American Society on Aging; all rights reserved.
    justification: The reference confirms that the VHA polytrauma system provides outpatient rehabilitation services near the patient's home, aligning with the claim.
    answer: Entailment
    ##
    claim: ### Machine Condition Monitoring - **Predictive Maintenance**: ML algorithms are not effective in predicting future damages in technical machines, such as turbines and pumps, as they fail to accurately model and extrapolate damage mechanisms based on sensor data.
    reference: Many technical machines are instrumented. Temperatures, pressures, flow rates, vibrations and so on are measured and centrally archived. These data can be used to reliably predict future damages several days in advance. A self-learning mathematical method is used to do this, which models the machine and can extrapolate the damage mechanism into the future. Examples include turbines, pumps and catalytic reactors that will be treated in this paper.
    justification: The claim directly contradicts with the given reference by mentioning that "ML algorithms are not effective in predicting future damages" as the reference mentions quite opposite of it.
    answer: Contradiction
    ##
    claim: Energy consumption of twin-screw extruders cannot be accurately predicted through simulation calculations, which often leads to underutilization of torque and suboptimal machine performance .
    reference: COMPOUNDING, the energy consumption of twin-screw extruders can be determined in advance by simulation calculation. This allows full torque utilisation, so that compounders can maximise use of available machine performance potential. © Carl Hanser Verlag.
    justification: The claim is saying cannot, but the reference says can be, so the claim is just opposite to the reference given.
    answer: Contradiction
    ##
    claim: Human-Robot Interaction (HRI): Robots are ineffective as guides in environments such as museums, where they fail to interact with visitors and provide relevant information .
    reference: As a testbed for real-world experimentation on HRI and dynamic interaction models, this paper presents an autonomous robot system acting as guide in a German arts museum. The visitors' evaluation of this system is analyzed using a questionnaire and reveals issues for subsequent analysis of the real-time interaction.
    justification: The claim misinterprets the reference by mentioning that "robots are ineffective and they fail to interact with visitors", while the reference acknowledges "issues but does not explicitly mentions a complete failure".
    answer: Contradiction
    ##
    claim: Ensuring that chatbots can handle a wide range of queries and provide accurate, relevant information is crucial for their effectiveness .
    reference:  In this paper we learn how to manage a dialogue relying on discourse of its utterances. We define extended discourse trees, introduce means to manipulate with them, and outline scenarios of multi-document navigation to extend the abilities of the interactive information retrieval-based chat bot. We also provide evaluation results of the comparison between conventional search and chat bot enriched with the multi-document navigation.
    justification: The claim and reference both discusses about the effectiveness of chatbots but the claim "mentions about wide range of queries" which is not directly mentioned in the reference.
    answer: Unverifiable
    ##
    claim: Suggestions for waste disposal techniques for space stations, particularly the Chinese space station, are discussed, emphasizing the need for advanced waste management solutions .
    reference:  [3] The inspect technology of space garbage is analyzed, such as radar, laser, lidar and so on. With STK, the move contrail of space garbage is also simulated. Finally, the future of the space garbage is assumed.    
    justification: The reference discusses space debris tracking technologies, not waste disposal inside space stations, making the claim unrelated and unverifiable based on that source.
    answer: Unverifiable
    ##
    claim: Origins and Histology: Ewing sarcoma: This type of cancer arises from neural crest cells. It is characterized by the production of small round blue cells by malignant cells .
    reference:  Osteosarcoma and Ewing sarcoma are the most common bone malignancies that affect children and adolescents, with an incidence of six new cases/1,000,000 inhabitants/year, accounting for approximately 7% of cancer diagnoses. They may be defined as neoplastic diseases that involve the bone tissues, the former arising from the mesenchymal bone forming cells and the latter from the neural crest cells.
    justification: no mention of round blue cells
    answer: Unverifiable
    ##
    Your response only answer 'Entailment' or 'Contradiction' or 'Unverifiable'
    Input:
    """

    client = genai.Client(api_key="")

    config = types.GenerateContentConfig(
        temperature=0
    )

    full_prompt = f"{prompt_prefix} \n claim: {claim} \n reference: {reference}\n answer:"
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
    df = pd.read_csv('../Data/df_1.csv')

    predictions = []

    for idx, row in df.iterrows():
        claim = row['claim_clean']
        reference = row['reference_clean']
        prediction = get_prediction(claim, reference, model_name)
        predictions.append(prediction)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}...")

        time.sleep(sleep_time)

    df['predict'] = predictions

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference clean data using Gemini.")
    parser.add_argument('--model', type=str, default = 'gemini-2.5-flash-preview-04-17', help='Model name to use (e.g., "models/gemini-1.5-pro-latest")')
    parser.add_argument('--output', type=str, default = './Result/gem2_3.csv', help='Path to output CSV file')
    parser.add_argument('--sleep_time', type=float, default=6.1, help='Sleep time (in seconds) between requests')
    main(parser.parse_args().output, parser.parse_args().sleep_time, parser.parse_args().model)
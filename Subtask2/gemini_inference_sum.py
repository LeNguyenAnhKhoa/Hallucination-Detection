import argparse
from google import genai
from google.genai import types
import pandas as pd
import time

def get_prediction(text, model_name):
    prompt_prefix = """
    You are an annotator concerned that the claim may not align with the reference. Your task is to determine whether the reference entail, is unrelated and unverifiable, is related but unverifiable, misinterpret, omit critical information, contain a numeric error, contain an opposite meaning, or contain an entity error to the claim.
    You will be given two inputs: claim, reference.
    You are asked to evaluate the generated text looking at the input text and the target text.
    A reference entails a claim if the information in the reference directly supports the claim being true and there is no conflicting information. The information logically leads to the claim being correct. A claim is categorized as unrelated and unverifiable if it discusses content that is completely unrelated to the reference abstracts and cannot be verified based on them. A claim is considered related but unverifiable when it remains topically relevant to the reference abstracts but still cannot be verified. Misinterpretation occurs when the claim presents logical fallacies, flawed reasoning such as over-claiming, under-claiming, ambiguity, inconsistency, or draws illogical conclusions compared to the reference. Missing information refers to cases where the claim omits critical parts of the reference abstracts, thereby altering the intended meaning or message. A numeric error happens when the claim includes a wrong numerical value that does not match the information in the abstracts. Opposite classification is used when the claim negates parts of the reference abstracts or replaces key terms with their antonyms, reversing the original meaning. Lastly, an entity error occurs when the claim incorrectly states an entity, such as a wrong object, concept, or name, compared to the reference abstracts.
    Only answer 'Opposite meaning' or 'Misrepresentation' or 'Related but unverifiable' or 'Entailment' or 'Entity error' or 'Unrelated and unverifiable' or 'Numeric error' or 'Missing information'.
    Example:
    ##
    claim: ### Machine Condition Monitoring - **Predictive Maintenance**: ML algorithms are not effective in predicting future damages in technical machines, such as turbines and pumps, as they fail to accurately model and extrapolate damage mechanisms based on sensor data .
    reference: Many technical machines are instrumented. Temperatures, pressures, flow rates, vibrations and so on are measured and centrally archived. These data can be used to reliably predict future damages several days in advance. A self-learning mathematical method is used to do this, which models the machine and can extrapolate the damage mechanism into the future. Examples include turbines, pumps and catalytic reactors that will be treated in this paper.
    answer: Opposite meaning
    ##
    claim: Advantages of Dropdown Menus: While dropdown menus can help avoid format errors, they are the only effective solution for fields requiring specific input formats, such as dates .
    reference: When an interactive form in the world wide web requires users to fill in exact dates, this can be implemented in several ways. This paper discusses an empirical online study with n = 172 participants which compared six different versions to design input fields for date entries. The results revealed that using a drop-down menu is best when format errors must be avoided, whereas using only one input field and placing the format requirements left or inside the text box led to faster completion time and higher user satisfaction. Copyright © 2011 Javier A. Bargas-Avila et al.
    answer: Misrepresentation
    ##
    claim: Ensuring that chatbots can handle a wide range of queries and provide accurate, relevant information is crucial for their effectiveness .
    reference: In this paper we learn how to manage a dialogue relying on discourse of its utterances. We define extended discourse trees, introduce means to manipulate with them, and outline scenarios of multi-document navigation to extend the abilities of the interactive information retrieval-based chat bot. We also provide evaluation results of the comparison between conventional search and chat bot enriched with the multi-document navigation.
    answer: Related but unverifiable
    ##
    claim: 8. X-ray Techniques: X-ray Absorption and Diffraction: Methods like extended X-ray absorption fine structure, X-ray diffraction, and low-angle scattering are used to gain and analyze experimental data for material examination .
    reference: Ways to gain and analyze experimental data obtained by X-ray techniques used in material examination are described. Emphasis is on the methods of extended X-ray absorption fine structure, X-ray diffraction, and X-ray low-angle scattering.
    answer: Entailment
    ##
    claim: Current State and Background of Cutting-Edge Fire Detection and Emergency Response Systems: Key Technologies and Approaches: Wireless Communication: LoRa Technology: Used in systems where temperature and flame sensors transmit data wirelessly to a central receiver, which processes the data and provides early warnings through a user-friendly GUI .
    reference: Fire detection systems are designed to discover fires and allow the safe evacuation of occupants as well as protecting the safety of emergency response personnel. This paper describes the design and development of a fire detection and alert system. Temperature and flame sensors are used to indicate the occurrence of fire. This work consists of two parts, which are transmitter and receiver, both using ZigBee wireless technology. Arduino Uno is used as the microcontroller at the transmitter part to control the sensor nodes and give alert when over temperature and flame are detected. At the transmitter, the collected data from the sensors are transmitted by an XBee module operated as router node. At the receiver side, an XBee coordinator module which is attached to a computer using USB to serial communication captured the data for further processing. In addition, an interactive and user-friendly Graphical User Interface (GUI) is developed. LabVIEW software is used to design the GUI which displays and analyze the possibility of fire happening. The system can display the fire location and provides early warning to allow occupants to escape the building safely.
    answer: Entity error
    ##
    claim: Mitigation Contributions: Dynamic Thresholds: Adaptation of dynamic thresholds for detecting low-rate DDoS attacks has shown high detection rates and low false-positive rates .
    reference: [13] To achieve a widespread deployment of Software-Defined Networks (SDNs) these networks need to be secure against internal and external misuse. Yet, currently, compromised end hosts, switches, and controllers can be easily exploited to launch a variety of attacks on the network itself. In this work we discuss several attack scenarios, which - although they have a serious impact on SDN - have not been thoroughly addressed by the research community so far. We evaluate currently existing solutions against these scenarios and formulate the need for more mature defensive means.
    answer: Unrelated and unverifiable
    ##
    claim: It is recommended that individuals consume at least 1000 mg of calcium and 10-20 micrograms of vitamin D daily to support bone health and maximize the effects of osteoporosis drug therapy .
    reference: A significant point of nutritional care and management for osteoporosis is that calcium and vitamin D are recommended to be actively administered on top of sufficient intake of energy and the other nutrients including protein. Daily intake of calcium and vitamin D is encouraged at least 800 mg and 10 to 20 microg, respectively. Calcium and vitamin D are also important for maximizing the effect of osteoporosis drug therapy. Supplement of calcium or vitamin D could be a supportive measure, when their necessary amount is difficult to be consumed.
    [2]: Physiological role of calcium and vitamin D in normal structure and metabolic regulation of bone tissue was presented. Calcium and vitamin D importance in their deficiencies supplementation, and in the prevention and therapy of osteoporosis was emphasized. Recommended calcium intake in different age groups and calcium content in selected salts were given.
    answer: Numeric error
    ##
    claim: Notable Drought Events: Southern Europe: The droughts in 2011/2012 were among the most extensive, affecting large areas and causing substantial ecological impacts .
    reference: A correct identification of drought events over vegetated lands can be achieved by detecting those soil moisture conditions that are both unusually dry compared with the 'normal' state and causing severe water stress to the vegetation. In this paper, we propose a novel drought index that accounts for the mutual occurrence of these two conditions by means of a multiplicative approach of a water deficit factor and a dryness probability factor. The former quantifies the actual level of plant water stress, whereas the latter verifies that the current water deficit condition is unusual for the specific site and period. The methodology was tested over Europe between 1995 and 2012 using soil moisture maps simulated by Lisflood, a distributed hydrological precipitation-runoff model. The proposed drought severity index (DSI) demonstrates to be able to detect the main drought events observed over Europe in the last two decades, as well as to provide a reasonable estimation of both extension and magnitude of these events. It also displays an improved adaptability to the range of possible conditions encountered in the experiment as compared with currently available indices based on the sole magnitude or frequency. The results show that, for the analyzed period, the most extended drought events observed over Europe were the ones in Central Europe in 2003 and in southern Europe in 2011/2012, while the events affecting the Iberian Peninsula in 1995 and 2005 and Eastern Europe in 2000 were among the most severe ones. © 2015 European Commission - Joint Research Centre. Hydrological Processes published by John Wiley & Sons Ltd.
    answer: Missing information
    ##
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
    parser.add_argument('--model', type=str, default = 'gemini-2.0-flash', help='Model name to use (e.g., "models/gemini-1.5-pro-latest")')
    parser.add_argument('--output', type=str, default = './Result/gem_sum_result.csv', help='Path to output CSV file')
    parser.add_argument('--sleep_time', type=float, default=4.2, help='Sleep time (in seconds) between requests')
    parser.add_argument('--path', type=str, default = '../Data/gem20fl_sum.csv', help='Path to input CSV file')
    main(parser.parse_args().output, parser.parse_args().sleep_time, parser.parse_args().model, parser.parse_args().path)
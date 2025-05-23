import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from groq import Groq

# Load the real data
data_csv = "data/real_data/Insurance/insurance.csv"
data = pd.read_csv(data_csv)

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.8, random_state=42)
test_data.to_csv('data/real_data/Insurance/insurance_test.csv', index=True)
train_data.to_csv('data/real_data/Insurance/insurance_train.csv', index=False) # Will include the entire training data, which will then be sampled in n sample sizes below.

# Define the n sample size of train_data
train_data = train_data.sample(200)

# Define temperature parameter for model (controls randomness and diversity, as temp -> 0, model becomes more deterministic and repetitive)
temperature = 1

# Load environment variables from the .env file
load_dotenv()

# Access the Groq API key (should be securely stored in the .env file (not provided, must be generated by user))
api_key = os.getenv("GROQ_API_KEY")

# Instantiate the Groq client with API key
client = Groq(api_key=api_key)

# List of model ID names that will be deployed. Visit groq API documentation for more models
model_names = ["llama-3.3-70b-versatile"]

# This prompt structure is adapted from the prompt example B.5. from the research paper: "Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in low-data regimes" (Seedatk, Huynh, et al.) https://arxiv.org/pdf/2312.12112 
# The template is currently adapted to the 'insurance.csv' dataset (referenced in README.md)
prompt_template_baseline = """
System role: You are a tabular synthetic data generation model.

You are a synthetic data generator.
Your goal is to produce data which mirrors the given examples in causal structure and feature and label distributions but also produce as diverse samples as possible.

I will give you real examples first.

Context: Leverage your knowledge about health, demographics, and insurance to generate 1000 realistic but diverse samples. 
Output the data in a csv format where I can directly copy and paste into a csv.

Example data: {data}

The output should use the following schema:

"age": integer // feature column for the person's age
"sex": string // feature column, male or female
"bmi": float // feature column for body mass index
"children": integer // feature column for number of children
"smoker": string // feature column, yes or no for smoking status
"region": string // feature column for region (northeast, southeast, southwest, northwest)
"charges": float // label column for insurance charges

DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
"""

prompt_template_advanced = """
System role: You are a tabular synthetic data generation model.

You are a synthetic data generator.
Your goal is to produce data which mirrors the given examples in causal structure and feature and label distributions but also produce as diverse samples as possible.

I will give you real examples first.

Context: Leverage your knowledge about health, demographics, and insurance to generate 200 realistic but diverse samples. 
Output the data in a JSON format.

Example data: {data}

The output should use the following schema:

"age": integer // feature column for the person's age
"sex": string // feature column, possible values: "male", "female"
"bmi": float // feature column for body mass index
"children": integer // feature column for the number of children
"smoker": string // feature column, possible values: "yes", "no"
"region": string // feature column for region, possible values: "northeast", "southeast", "southwest", "northwest"
"charges": float // target label column for insurance charges

Here are detailed summary stats that you should also use:

,count,unique,top,freq,mean,std,min,25%,50%,75%,max
age,1338.0,,,,39.20702541106129,14.049960379216154,18.0,27.0,39.0,51.0,64.0
sex,1338,2,male,676,,,,,,,
bmi,1338.0,,,,30.66339686098655,6.098186911679014,15.96,26.29625,30.4,34.69375,53.13
children,1338.0,,,,1.0949177877429,1.205492739781914,0.0,0.0,1.0,2.0,5.0
smoker,1338,2,no,1064,,,,,,,
region,1338,4,southeast,364,,,,,,,
charges,1338.0,,,,13270.422265141257,12110.011236694001,1121.8739,4740.28715,9382.033,16639.912515,63770.42801

DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
"""

# Function to generate synthetic data using a model and prompt
def generate_synthetic_data(model_name, data):

    prompt = prompt_template_advanced.format(data = data)
    
    try:
        # Create a chat completion using the Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model=model_name,
            response_format={"type": "json_object"}, #Turn on for JSON beta mode
            temperature=temperature
        )
        
        # Print the full response for debugging
        print("Full Response:", response)
        
        generated_data = response.choices[0].message.content if response.choices else "No output"
        
        return generated_data
    except Exception as e:
        print(f"Error generating data with model {model_name}: {e}")
        return None

# Main function to run the process
def main():
    
    for model_name in model_names:
        print(f"Generating data with {model_name}...")
        
        # Generate synthetic data with n rows!
        data = generate_synthetic_data(model_name, train_data)

        print(f"Generated Data for {model_name}:\n{data}\n")

if __name__ == "__main__":
    main()
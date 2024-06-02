import pandas as pd
import os
import json
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker, OpenAIAgent
from llama_index.core.tools import FunctionTool
from openai import OpenAI as OpenAI
from llama_index.llms.openai import OpenAI as LLMOpenAI

# Load the school data
try:
    school_data = pd.read_excel('sampledata.xlsx')
    print("School data loaded successfully")
except Exception as e:
    print(f"Error loading school data: {e}")
    school_data = pd.DataFrame()  # Load an empty dataframe or handle error appropriately

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_criteria_with_gpt4(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract criteria for school selection from the given prompt and return it in the following JSON format: {\"max_tuition\": number, \"religious_affiliation\": boolean, \"math_sat\": number, \"reading_sat\": number, \"state\": string, \"distance\": number, \"place_distance_is_from\": string}. Ensure the output is a valid JSON object without any additional text. Use state abbreviations."},
                {"role": "user", "content": prompt}
            ]
        )
        criteria_json = response.choices[0].message.content.strip()
        
        try:
            criteria = json.loads(criteria_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            criteria = {}
        
        return criteria
    except Exception as e:
        print(f"Error in extract_criteria_with_gpt4: {e}")
        return {}

def categorize_score(score, row, test_type):
    quartiles = [25, 50, 75]
    for quartile in quartiles:
        if score < row[f'{test_type}{quartile}']:
            return f"Below {quartile}th percentile"
    return "Above 75th percentile"

def filter_by_distance(data, place_distance_is_from, max_distance):
    geolocator = Nominatim(user_agent="school_locator")
    location = geolocator.geocode(place_distance_is_from)
    if not location:
        print(f"Error: Unable to locate {place_distance_is_from}")
        return data
    
    origin_coords = (location.latitude, location.longitude)

    def calculate_distance(row):
        school_coords = (row['LATITUDE'], row['LONGITUDE'])
        return geodesic(origin_coords, school_coords).miles

    data['Distance'] = data.apply(calculate_distance, axis=1)
    return data[data['Distance'] <= max_distance]

def filter_schools(criteria):
    filtered_data = school_data.copy()

    if 'max_tuition' in criteria:
        filtered_data = filtered_data[filtered_data['tuition'] <= criteria['max_tuition']]

    if 'religious_affiliation' in criteria:
        if criteria['religious_affiliation']:
            filtered_data = filtered_data[filtered_data['Religious_Affiliation'] != -2]
        else:
            filtered_data = filtered_data[filtered_data['Religious_Affiliation'] == -2]

    if 'math_sat' in criteria:
        filtered_data['Math_SAT_Category'] = filtered_data.apply(lambda row: categorize_score(criteria['math_sat'], row, 'SAT_Math'), axis=1)
        filtered_data = filtered_data[filtered_data['Math_SAT_Category'] == 'Above 75th percentile']
        
    if 'reading_sat' in criteria:
        filtered_data['Reading_SAT_Category'] = filtered_data.apply(lambda row: categorize_score(criteria['reading_sat'], row, 'SAT_Reading'), axis=1)
        filtered_data = filtered_data[filtered_data['Reading_SAT_Category'] == 'Above 75th percentile']

    if 'distance' in criteria and 'place_distance_is_from' in criteria:
        filtered_data = filter_by_distance(filtered_data, criteria['place_distance_is_from'], criteria['distance'])

    return filtered_data

def generate_response(filtered_schools):
    if not isinstance(filtered_schools, pd.DataFrame) or filtered_schools.empty:
        return "No schools found matching the criteria."

    top_schools = filtered_schools.head(10)
    school_names = top_schools['School_Name'].tolist()

    response = f"Here are 10 choices for schools you are very likely to get into based on your criteria:\n"
    for i, school in enumerate(school_names, 1):
        response += f"{i}. {school}\n"

    return response

# Define FunctionTool instances
tool_extract_criteria = FunctionTool.from_defaults(fn=extract_criteria_with_gpt4)
tool_filter_schools = FunctionTool.from_defaults(fn=filter_schools)
tool_generate_response = FunctionTool.from_defaults(fn=generate_response)

# Create a list of tools
tools = [
    tool_extract_criteria,
    tool_filter_schools,
    tool_generate_response
]

# Initialize the agent
llm = LLMOpenAI(model="gpt-4o")
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# Example usage of agent communication
prompt = "Given that my Math SAT is 530, my SAT reading is 480 and I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?"

criteria = extract_criteria_with_gpt4(prompt)
filtered_schools = filter_schools(criteria)
response = generate_response(filtered_schools)
print(response)

response =agent.chat("Hi")
print(response)
response =agent.chat("Given that my Math SAT is 730, my SAT reading is 780 and I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?")
print(response)

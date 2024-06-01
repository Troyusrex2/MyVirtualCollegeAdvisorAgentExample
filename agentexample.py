import pandas as pd
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker, OpenAIAgent
from llama_index.core.tools import FunctionTool
from openai import OpenAI as OpenAI
from llama_index.llms.openai import OpenAI as LLMOpenAI

# Load the data
try:
    school_data = pd.read_excel('sampledata.xlsx')
    print("School data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    school_data = pd.DataFrame()  # Load an empty dataframe or handle error appropriately

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_criteria_with_gpt4(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract criteria for school selection from the given prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    criteria = response.choices[0].message.content.strip()
    
    try:
        return eval(criteria)
    except Exception as e:
        print(f"Error parsing criteria: {e}")
        return {}

def categorize_score(score, row, test_type):
    quartiles = [25, 50, 75]
    for quartile in quartiles:
        if score < row[f'{test_type}{quartile}']:
            return f"Below {quartile}th percentile"
    return "Above 75th percentile"

def filter_by_distance(data, origin, max_distance):
    geolocator = Nominatim(user_agent="school_locator")
    location = geolocator.geocode(origin)
    if not location:
        return data
    
    origin_coords = (location.latitude, location.longitude)

    def calculate_distance(row):
        school_coords = (row['latitude'], row['longitude'])
        return geodesic(origin_coords, school_coords).miles

    data['distance'] = data.apply(calculate_distance, axis=1)
    return data[data['distance'] <= max_distance]

def filter_schools(criteria):
    filtered_data = school_data

    if 'max_tuition' in criteria:
        filtered_data = filtered_data[filtered_data['tuition'] <= criteria['max_tuition']]

    if 'religious_affiliation' in criteria and criteria['religious_affiliation']:
        filtered_data = filtered_data[filtered_data['rel_affil'] != -2]
    elif 'religious_affiliation' in criteria and not criteria['religious_affiliation']:
        filtered_data = filtered_data[filtered_data['rel_affil'] == -2]

    if 'math_sat' in criteria:
        filtered_data['Math_SAT_Category'] = filtered_data.apply(lambda row: categorize_score(criteria['math_sat'], row, 'math_sat'), axis=1)
    if 'reading_sat' in criteria:
        filtered_data['Reading_SAT_Category'] = filtered_data.apply(lambda row: categorize_score(criteria['reading_sat'], row, 'reading_sat'), axis=1)

    if 'math_sat' in criteria:
        filtered_data = filtered_data[filtered_data['Math_SAT_Category'] == 'Above 75th percentile']
    if 'reading_sat' in criteria:
        filtered_data = filtered_data[filtered_data['Reading_SAT_Category'] == 'Above 75th percentile']

    if 'state' in criteria:
        filtered_data = filtered_data[filtered_data['state'] == criteria['state']]

    if 'distance' in criteria and 'place_distance_is_from' in criteria:
        filtered_data = filter_by_distance(filtered_data, criteria['place_distance_is_from'], criteria['distance'])

    return filtered_data

def generate_response(filtered_schools):
    top_schools = filtered_schools.head(10)
    school_names = top_schools['name'].tolist()

    response = f"Here are your top 10 choices for schools you are very likely to get into based on your criteria:\n"
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
prompt = "Given that my Math SAT is 530, my SAT reading is 480 and I wish to go to an academic university plus I can only afford up to $38,000 a year in tuition. I would prefer a religiously affiliated school in California within 50 miles of Santa Clara. Where are my top ten choices for schools I am very likely to get in to?"

response = agent.chat(prompt)
print(response)

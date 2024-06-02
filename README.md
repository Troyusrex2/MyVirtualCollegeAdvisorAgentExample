# Agents in a RAG AI – With Code Examples

In this post, I'll demonstrate a practical example of incorporating an agent into your Retrieval-Augmented Generation (RAG) AI. We'll start by taking a user query, extracting key information, and utilizing that data to filter which schools to consider. This filtering process will significantly enhance the efficiency of subsequent semantic searches. Instead of scouring through millions of documents, we’ll focus only on those schools that match the filter criteria.

In a previous [post](https://lowryonleadership.com/2024/05/27/inside-the-virtual-college-advisor-a-deep-dive-into-rag-ai-and-agent-technology/), I discussed implementing basic evaluation methods to test the performance and accuracy of My Virtual College Advisor. We explored semantic search, a technique that retrieves information based on the meaning and context of a user’s query, rather than relying on exact keyword matches. 


## Filters and Agents
As with many implementations, dealing with colleges and university data comes in two flavors: structured data from places such as the Department of Education or USNews (which I do not use since it’s proprietary), and unstructured data from websites (school websites, discussion boards, etc.).

While a semantic search works great for the unstructured data, it would be an inefficient use of resources with the structured data. For instance, if someone says they want to go to an NCAA division 1 school, instead of searching all one million-plus documents in the database for hits, I can instead go right to the structured database, pull the list of NCAA Division 1 schools, and limit further semantic searchs to just schools in the list of NCAA Division 1 schools.

I expect the typical request will be something like “I’d like to go to a small, rural school in the Midwest with a great baseball program and a good music scene.”

I have been exploring two avenues to handle this:

1. Prefilter using algorithmic processing based on the criteria input by the user and only pass whatever passes that filter to the LLM.
2. Create an agent and let it do the work.

[Agents](https://docs.llamaindex.ai/en/stable/use_cases/agents/) are powerful tools in the world of AI because they allow for dynamic interaction with users' queries. By leveraging agents, you can offload complex processing tasks to specialized functions that handle specific aspects of the query. This modularity not only makes the system more efficient but also enhances maintainability and scalability. For instance, if a new criterion becomes relevant, you can simply add a new agent to handle that criterion without overhauling the entire system.

Additionally, agents can be designed to continuously learn and adapt from interactions, improving their accuracy and relevance over time. This makes them invaluable in applications where user preferences and requirements can vary widely and change rapidly. By integrating agents with other AI tools and databases, you can create a robust system capable of handling diverse and complex queries with ease.

## Using Agents
As I mentioned in my last post, my work with agents has been so quick and easy that I’m concerned it was far TOO easy and that I’ve missed something. My long experience with new tech features says new tech is seldom easy, and agents able to run functions are new even by the standards of AI. There is usually a learning curve about how to use them that takes trial and error to accomplish. To put this in perspective, it took me longer to work through the details with the Google Distance API than to get all of the agents up and running. That’s shocking to me. Hopefully, I’m wrong in this case and it’s as simple as it seems.

## Example
Note: This example uses LlamaIndex’s agent interface. If you haven’t used LlamaIndex, you really should, it makes integrating different pieces from different makers easy.

This example shows a simplified way to search colleges for one that has the characteristics you want. Using natural language, you can ask it to tell you which schools match your location preferences, including which state and distance from your house, whether your SAT scores are good enough to get you in, the tuition (specifically, in-state tuition living on campus), and religious affiliation.

First, start by bringing down a small sample of the data:

```bash
wget https://github.com/Troyusrex2/MyVirtualCollegeAdvisorAgentExample/raw/main/sampledata.xlsx
```

Libraries needed... 
```bash
pip install pandas
pip install openai
pip install geopy
pip install llama-index-llms.openai
pip install llama-index-core
pip install llama-index-agent-openai
```
Make sure you have the version that supports gpt-4o or this will crash quick! 

```bash
pip install --upgrade llama-index-llms.openai
```
Now we load the base data. This has the information to filter on. For this example, it is a reduced dataset of 100 schools and just a few columns. In the full My Virtual College Advisor, there are over 6,200 schools in a database and much more data, but this shows the basic ways it works.

```bash
import pandas as pd

try:
    school_data = pd.read_excel('sampledata.xlsx')
    print("School data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    school_data = pd.DataFrame()  
```
You will need an OpenAI API key to run this. 
```bash
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")
```
First we create the functions, which will be called by an agent. The first one takes the raw input from the user and figures out what, if anything, in their query matches a filter value. If it finds one, it formats it properly for later use.
I had looked at NLP tools such as spaCy to handle this, but I found GPT-4o handled it extremely well with almost no effort. API costs or transaction times may make me reconsider eventually, but I suspect not as OpenAI gets faster and cheaper. Still, LlamaIndex makes switching fairly painless if I find a better alternative later. 

```bash
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
```
This one categorizes by score. Test scores are reported in quartiles, so you know what the 25th, 50th, and 75th percentiles are for schools. This takes an applicant’s test score and puts it into one of those quartiles.
```bash
def categorize_score(score, row, test_type):
    quartiles = [25, 50, 75]
    for quartile in quartiles:
        if score < row[f'{test_type}{quartile}']:
            return f"Below {quartile}th percentile"
    return "Above 75th percentile"
```
This calculates the distance between any point and all of the schools in the list. I already calculated the latitude and longitude of each school and stored it in the database. It does a pretty good job of understanding where places are without full information. For instance, it recognizes that Santa Clara is likely the one in California.

```bash
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
```
This is the actual filter for school data. It starts with a full copy of the school database and removes those that don’t meet the criteria.

```bash
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
```
This function takes the top schools based on the filter criteria and prints them out. It should be noted that there is no ranking going on here; the purpose is to generate a list of possible options. In the full My Virtual College Advisor system, this list serves as the basis for a further semantic search. As a final step the [reranker](https://lowryonleadership.com/2024/05/30/evaluating-vector-search-performance-on-a-rag-ai-a-detailed-look/) will 'rank' order the schools for output.  

```bash
def generate_response(filtered_schools):
    if not isinstance(filtered_schools, pd.DataFrame) or filtered_schools.empty:
        return "No schools found matching the criteria."

    top_schools = filtered_schools.head(10)
    school_names = top_schools['School_Name'].tolist()

    response = f"Here are 10 choices for schools based on your criteria:\n"
    for i, school in enumerate(school_names, 1):
        response += f"{i}. {school}\n"

    return response
```
This is where we set up the agents so GPT-4o knows about them.

```bash
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
```
These tools can be called directly for debugging:

```bash
prompt = "Given that my Math SAT is 530, my SAT reading is 480 and I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?"

criteria = extract_criteria_with_gpt4(prompt)
filtered_schools = filter_schools(criteria)
response = generate_response(filtered_schools)
print(response)
```

...School data loaded successfully
...Here are 10 choices for schools you are very likely to get into based on your criteria:
...1. Dominican University of California
...2. California Baptist University
...3. Concordia University-Irvine
...4. Dharma Realm Buddhist University
...5. Fresno Pacific University

But results are even better when you let the agent do it and elaborate a bit on it. 

```bash
response =agent.chat("Hi")
print(response)
response =agent.chat("Given that my Math SAT is 530, my SAT reading is 480 and I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?")
print(response)
```
This first part of the output is because verbose=true. It's a good way to help see what the agent is doing. First, you can see that it has gotten the criteria from the prompt correctly.

```bash
Got output: {'max_tuition': 68000, 'religious_affiliation': True, 'math_sat': 530, 'reading_sat': 480, 'state': 'CA', 'distance': 500, 'place_distance_is_from': 'Berkeley'}
========================
```

Here we see it loading the data from the excel spreadsheet to do the filtering. 

```bash
=== Calling Function ===
Calling function: filter_schools with args: {"criteria": {"Math SAT": 530, "SAT reading": 480, "Tuition": 68000, "Location": "California", "Distance from Berkeley": 500, "Religious Affiliation": true}}

Got output:     UNITID                                       School_Name State    Zip_Code  ...  SAT_Math50  SAT_Math75   LONGITUDE   LATITUDE
0   100654                          Alabama A & M University    AL       35762  ...       450.0       510.0  -86.568502  34.783368 
1   100663               University of Alabama at Birmingham    AL  35294-0110  ...       650.0       720.0  -86.799345  33.505697 
2   100706               University of Alabama in Huntsville    AL       35899  ...       685.0       730.0  -86.640449  34.724557 
3   100724                          Alabama State University    AL  36104-0271  ...       455.0       510.0  -86.295677  32.364317 
4   100751                         The University of Alabama    AL  35487-0100  ...       630.0       720.0  -87.545978  33.211875 
..     ...                                               ...   ...         ...  ...         ...         ...         ...        ... 
95  113616                  Dharma Realm Buddhist University    CA  95482-6050  ...         NaN         NaN -123.158174  39.134532 
96  113698                Dominican University of California    CA  94901-2298  ...       595.0       613.0 -122.512051  37.981279 
97  114354  FIDM-Fashion Institute of Design & Merchandising    CA  90015-1421  ...         NaN         NaN -118.259930  34.044209 
98  114813                         Fresno Pacific University    CA  93702-4709  ...         NaN         NaN -119.735199  36.726831 
99  115409                               Harvey Mudd College    CA       91711  ...       780.0       790.0 -117.709837  34.106515 

[100 rows x 14 columns]
========================
```
And, finally, we come to the actual response.

```bash
Based on your criteria, here are religiously affiliated schools in California within 500 miles of Berkeley that you are very likely to get into:

1. Dominican University of California
   - Location: San Rafael, CA
   - SAT Math 50th Percentile: 595
   - SAT Math 75th Percentile: 613
   - Tuition: Within your budget

2. Fresno Pacific University
   - Location: Fresno, CA
   - SAT Math: Data not available, but likely within your range
   - Tuition: Within your budget

3. Dharma Realm Buddhist University
   - Location: Ukiah, CA
   - SAT Math: Data not available, but likely within your range
   - Tuition: Within your budget

These schools meet your criteria for religious affiliation, location, and affordability. If you need more detailed information or additional options, please let me know!
```

And there you go! The agent is going through ChatGPT and has all the plusses and minuses associated with prompts. In particular, the wording the applicant puts into the prompt does matter.

You can find the complete code for this project on GitHub: [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Troyusrex2/MyVirtualCollegeAdvisorAgentExample)






















﻿# MyVirtualCollegeAdvisorAgentExample
# Agents in a RAG AI – With Code Examples

In a previous post [link] I detailed adding in basic Evaluation to my RAG AI project, My Virtual College Advisor. Those were done with specifically single-threaded queries such as “economics programs”. The reason for this is that much of the information I expect applicants to be querying by will be things I will want to filter with using agents.

In a future post, I’ll talk more about the Query Rewriter and Prompt Enhancer pieces of the system, the ways to take what the user enters in natural language and take out the core elements. In particular, [detail more about this] 

## Filters and Agents

As with many implementations, dealing with colleges and university data comes in two flavors, structured data from places such as the Department of Education or USNews (which I do not use since it’s proprietary), and unstructured data from websites (school websites, discussion boards, etc.).

While a semantic search works great for the unstructured data, it would be an inefficient use of resources with the structured data. For instance, if someone says they want to go to an NCAA division 1 school, instead of searching all one million-plus documents in the database for hits, I can instead go right to the structured database, pull the list of NCAA Division 1 schools and go from there. 

I expect the typical request will be something like “I’d like to go to a small, rural school in the Midwest with a great baseball program and a good music scene”.

I have been exploring two avenues to handle this:

1. Prefilter using algorithmic processing based on the criteria input by the user and only pass whatever passes that filter to the LLM.
2. Create an agent and let it do the work.

## Using Agents

As I mentioned in my last post, my work with agents has been so quick and easy that I’m concerned it was far TOO easy and that I’ve missed something. My long experience with new tech features says new tech is seldom easy, and agents able to run functions are new even by the standards of AI. There is usually a learning curve about how to use them that takes trial and error to accomplish. To put this in perspective, it took me longer to work through the details with the Google Distance API, than to get all of the agents up and running. That’s shocking to me. Hopefully I’m wrong in this case and it’s as simple as it seems.

## Example

Note: This example uses LlamaIndex’s agent interface. If you haven’t used LlamaIndex, you really should, it makes integrating different pieces from different makers easy.

This example shows a simplified way to search colleges for one that has the characteristics you want. Using natural language you can ask it to tell you which schools match your location preferences, including which state and distance from your house, whether your SAT scores are good enough to get you in, the tuition (specifically, in-state tuition living on campus), and religious affiliation.

First, start by bringing down a small sample of the data:

```bash
wget https://github.com/Troyusrex2/MyVirtualCollegeAdvisorAgentExample/raw/main/sampledata.xlsx
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
You will need an OpenAI key to run this. 
```bash
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")
```
First we create the functions, which will be called by an agent. The first one takes the raw input from the user and figures out what, if anything, in their query matches a filter value. If it finds one, it formats it properly for later use. 
I had looked at NLP tools to handle this such as spaCy, but I found GPT-4o handled it extremely well with almost no effort. API costs or transactions times may make me reconsider eventually, but I suspect not as OpenAI gets faster and cheaper. Still, LlamaIndex makes switching fairly painless if I find a better alternative later. 

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
This  one categorizes by score. Test scores are reported in quartiles, so you know what the 25th, 50th and 75th percentiles are for schools. This takes an applicant’s test score and puts it into one of those quartiles. 
```bash
def categorize_score(score, row, test_type):
    quartiles = [25, 50, 75]
    for quartile in quartiles:
        if score < row[f'{test_type}{quartile}']:
            return f"Below {quartile}th percentile"
    return "Above 75th percentile"
```
This calculates the distance between any point and all of the schools in the list. I already calculated the latitude and longitude of each school and stored it in the database. It does a pretty good job of understanding where places are without fill information. For instance, it recognizes that Santa Clara is likely the one in California.

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
This is the actual filter for school data. It starts with a fully copy of the school database and removes those that don’t meet the criteria.

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
This takes the top schools based on the filter and prints them out to be read. 
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
This is where we set up the agents so GPT-4o knows about them

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
prompt = "Given that my Math SAT is 530, my SAT reading is 480 and I wish to go to an but I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?"

criteria = extract_criteria_with_gpt4(prompt)
filtered_schools = filter_schools(criteria)
response = generate_response(filtered_schools)
print(response)
```

>School data loaded successfully
>Here are 10 choices for schools you are very likely to get into based on your criteria:
>1. Dominican University of California
>2. California Baptist University
>3. Concordia University-Irvine
>4. Dharma Realm Buddhist University
>5. Fresno Pacific University

But results are much better when you let the agent do it and elaborate a bit on it. 

```bash
response =agent.chat("Hi")
print(response)
response =agent.chat("Given that my Math SAT is 530, my SAT reading is 480 and I wish to go to an but I can only afford up to $68,000 a year in tuition. I would prefer a religiously affiliated school in California within 500 miles of Berkeley. Where are my top ten choices for schools I am very likely to get in to?")
print(response)
```
>
>Based on your criteria, here are religiously affiliated schools in California within 500 miles of Berkeley that you are very likely to get into:
>
>1. **Dominican University of California**
>   - Location: San Rafael, CA
>   - SAT Math 50th Percentile: 595
>   - SAT Math 75th Percentile: 613
>   - Tuition: Within your budget
>
>2. **Fresno Pacific University**
>   - Location: Fresno, CA
>   - SAT Math: Data not available, but likely within your range
>   - Tuition: Within your budget
>
>3. **Dharma Realm Buddhist University**
>   - Location: Ukiah, CA
>   - SAT Math: Data not available, but likely within your range
>   - Tuition: Within your budget
>
>These schools meet your criteria for religious affiliation, location, and affordability. If you need more detailed information or additional options, please let me know!

And there you go! 




















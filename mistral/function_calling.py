"""
in this module we will be using mistral function calling to get flight details and book a flight
"""

import os
import json
from mistralai import Mistral
import functools
import pandas as pd

# adding dummy data for reference
# Create a dictionary with the data
data = {
    'Origin Airport': ['JFK', 'LHR', 'SYD', 'DEL', 'HND', 'YYZ', 'GRU', 'FRA', 'PEK', 'DXB'],
    'Destination Airport': ['LAX', 'CDG', 'MEL', 'BOM', 'NRT', 'YVR', 'GIG', 'MUC', 'SHA', 'AUH'],
    'Date': ['2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30'],
    'Time': ['08:00 AM', '10:30 AM', '02:45 PM', '05:15 PM', '07:30 AM', '09:45 AM', '01:00 PM', '03:20 PM', '06:50 AM', '08:55 PM'],
    'Flight Name': ['Delta 123', 'British Airways 456', 'Qantas 789', 'IndiGo 1011', 'ANA 1213', 'Air Canada 1415', 'LATAM 1617', 'Lufthansa 1819', 'Air China 2021', 'Emirates 2223']
}

# Create the DataFrame
df = pd.DataFrame(data)

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def get_flight_info(origin, destination):
    # Filter the DataFrame based on origin and destination
    result = df[(df['Origin Airport'] == origin) & (df['Destination Airport'] == destination)]

    # Check if any rows match the criteria
    if result.empty:
        return json.dumps({"error": "No flights found for the given origin and destination."})

    # Convert the result to a dictionary and then to JSON
    result_dict = result.to_dict(orient='records')[0]
    return json.dumps(result_dict, indent=4)

# defining a tools with function for getting flight info - just the origin and destination
tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "get_flight_info",
            "description" : "Get the desination and origin airport locations from the user query",
            "parameters" : {
                "type" : "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "flight origin eg. BLR"
                    },
                    "destination": {
                        "type": "string",
                        "description": "flight destination eg. BOM"
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    }
]

# functions_list = {
#     "get_flight_info": functools.partial()
# }

messages = [{"role": "user", "content": "when is the next flight from sydney to melbourne?"}]

response = client.chat.complete(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="any"
)

messages.append(response.choices[0].message)

# print(response.choices[0].message.tool_calls[0].function.arguments)
# tool_response = response.choices[0].message.tool_calls[0].function.arguments
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)
# function_params = json.loads(tool_response)
# print(function_name,function_params['origin'])

user_function_names = {
    "get_flight_info": functools.partial(get_flight_info)
}

user_function_calling_results = user_function_names[function_name](function_params['origin'], function_params['destination'])

# print(user_function_calling_results)

messages.append({"role":"tool", "name": function_name, "content":user_function_calling_results, "tool_call_id": tool_call.id})

# print("this is messages", messages)
second_response = client.chat.complete(model=model, messages=messages)

print(second_response.choices[0].message.content)
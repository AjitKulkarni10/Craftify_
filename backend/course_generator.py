from groq import Groq
import json
import re
import time
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("api_key")

if not api_key:
    raise ValueError("API key not found in Colab secrets. Please ensure that the 'GROQ_API_KEY' is added to the Colab secrets.")

def initialize_groq_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None
    
def course_creator_model(client, input_topic, difficulty_level, context=None):
    system_prompt = f"""
        You are an advanced AI designed to create comprehensive online courses from scratch. When given a topic and a difficulty level, generate a complete course structure with the following components:

        Course Title: Create a clear, engaging title for the course.
        Course Overview: Provide a brief introduction to the course, its objectives, and target audience.
        Difficulty Level: The course content should be tailored to the specified difficulty level: {difficulty_level}.

        1. Module Design:
            Identify Sub-Topics: Analyze the main topic and determine all relevant sub-topics that need to be covered.
            Module Breakdown: Divide the main topic into distinct modules, each focusing on one or more sub-topics.
    
        2. Chain of Thought for Each Sub-Topic:
            Conceptual Analysis: For each sub-topic, start by defining the core concepts and principles. Explain the importance and relevance of each concept.
            Detailed Explanation: Break down the concepts into fundamental components. Use simple language to explain complex ideas.
            Illustrative Examples: Provide multiple examples, case studies, or scenarios that apply the concepts in practical contexts.
            Comparative Analysis: Compare different theories or viewpoints related to the sub-topic, if applicable.
            Contextual Relevance: Explain how the sub-topic applies to real-world situations or problems.

        3. Content Quality:
            Ensure each module logically flows from one to the next.
            Maintain clarity and accuracy in all explanations.
            Structure the content to be beginner-friendly and suitable for online learning platforms.

        Ensure that the format will be JSON for ease of understanding and keep the indentation properly.

        Ensure that the course content progresses logically, building on previous modules, and uses accessible language for those new to the subject. Focus on delivering production-ready content suitable for online learning platforms.
    """

    user_prompt = f"""
    I want to design a course on {input_topic} for {difficulty_level} level.
    Please generate a complete, module-wise course content that is suitable for the given difficulty level.
        1. Break Down the Main Topic: Identify and list all relevant sub-topics.
        
        2. Detailed Sub-Topic Explanation:
        Define core concepts and principles.
        Break down concepts into fundamental components.
        Provide multiple examples, case studies, or practical scenarios.
        Include comparative analyses if relevant.
        Explain the real-world relevance of each concept.


        Provide the output only in the structure json format and do not provide any other text that the course content. Not a single extra line should be given in the output.

        Structure: The output should be a valid JSON object with the following structure:
            {{
                "courseTitle": "string",
                "courseOverview": "string",
                "modules": [
                    {{
                        "moduleTitle": "string",
                        "moduleOverview": "string",
                        "keyTopics": ["string"],
                        "detailedContent": [
                            {{
                                "concept": "string",
                                "explanation": "string",
                                "example": "string",
                                "realWorldRelevance": "string"
                            }}
                        ]
                    }}
                ]
            }}
        No other word except the course content should be present in the output.
    """

    chat_completion = client.chat.completions.create(
           messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
           ],
            model="llama3-70b-8192",
        )
    return chat_completion.choices[0].message.content

def take_user_input_and_create_course(topic, difficulty_level):
    print(f"Starting course generation for topic: {topic}, difficulty: {difficulty_level}")
    
    client = initialize_groq_client(api_key)
    if not client:
        print("Failed to initialize Groq client")
        raise ValueError("Model initialization failed. Please check the API and try again.")

    try:
        # Call the course creator function with the user's input
        course_content = course_creator_model(client, topic, difficulty_level)
        print("Raw course content received from model")

        if not course_content:
            print("Empty course content received")
            raise ValueError("Failed to generate the course content.")

        # Try to parse the content as JSON
        import json
        try:
            parsed_content = json.loads(course_content)
            print("Successfully parsed course content as JSON")
            return parsed_content
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Raw content:", course_content)
            raise ValueError(f"Invalid JSON format in course content: {str(e)}")

    except Exception as e:
        print(f"Error in course generation: {str(e)}")
        import traceback
        print("Full traceback:", traceback.format_exc())
        raise ValueError(f"Error generating course: {str(e)}")

def generate_mindmap_data(course_content):
    """
    Convert course content into a hierarchical mindmap structure.
    Returns a simplified format suitable for diagram visualization.
    """
    try:
        print("Starting mindmap generation with content:", course_content)
        
        # If course_content is a string, parse it as JSON
        if isinstance(course_content, str):
            import json
            try:
                course_content = json.loads(course_content)
                print("Successfully parsed JSON string")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON string: {str(e)}")
                raise ValueError(f"Invalid JSON format: {str(e)}")

        # Validate required fields
        if not isinstance(course_content, dict):
            raise ValueError("Course content must be a dictionary")
            
        if "courseTitle" not in course_content:
            raise ValueError("Course content missing 'courseTitle'")
            
        if "modules" not in course_content or not isinstance(course_content["modules"], list):
            raise ValueError("Course content missing or invalid 'modules' array")

        # Create the root node (course)
        mindmap_data = {
            "id": "root",
            "type": "root",
            "text": course_content["courseTitle"],
            "children": []
        }

        # Add modules as first-level children
        for idx, module in enumerate(course_content["modules"]):
            if not isinstance(module, dict):
                print(f"Warning: Module {idx} is not a dictionary, skipping")
                continue
                
            if "moduleTitle" not in module:
                print(f"Warning: Module {idx} missing 'moduleTitle', skipping")
                continue
                
            if "keyTopics" not in module or not isinstance(module["keyTopics"], list):
                print(f"Warning: Module {idx} missing or invalid 'keyTopics', using empty list")
                module["keyTopics"] = []

            module_node = {
                "id": f"module_{idx}",
                "type": "module",
                "text": module["moduleTitle"],
                "children": [
                    {
                        "id": f"topic_{idx}_{t_idx}",
                        "type": "topic",
                        "text": topic
                    }
                    for t_idx, topic in enumerate(module["keyTopics"])
                ]
            }
            mindmap_data["children"].append(module_node)

        print("Successfully generated mindmap data")
        return mindmap_data
        
    except Exception as e:
        print(f"Error generating mindmap data: {str(e)}")
        raise ValueError(f"Failed to generate mindmap: {str(e)}")

def quiz_creator_model(client, input_topic):
    
    # Simplified system prompt with strict JSON formatting
    system_prompt = """You are a quiz generator. Generate a quiz in strict JSON format only.
                        The JSON must be properly formatted with no trailing commas and all strings must be properly escaped.
                        IMPORTANT: 
                        1. Each question MUST have exactly 4 options (A, B, C, D)
                        3. No trailing commas
                        4. All strings must be in double quotes
                        5. All object properties must be strictly followed by a colon
                        6. All values must be properly formatted (strings in quotes, numbers without quotes)
                        7. No line breaks within string values
                        8. Start with { and end with }
                        9. Strictly use commas between all properties
                        10. No extra text before or after the JSON

                        Strictly follow the struture for the output:
                        Example format:
                        {
                            "quizTitle": "Basic Math Quiz",
                            "totalQuestions": 3,
                            "timeLimit": 15,
                            "questions": [
                                {
                                    "id": "q1",
                                    "question": "What is 2 + 2?",
                                    "options": [
                                        {"id": "A", "text": "3"},
                                        {"id": "B", "text": "4"},
                                        {"id": "C", "text": "5"},
                                        {"id": "D", "text": "6"}
                                    ],
                                    "correctOptionId": "B",
                                    "points": 2
                                }
                            ]
                        }
    """

    user_prompt = f"""Generate a quiz about {input_topic} with 10 questions.
Follow these rules strictly:
1. Return only valid JSON
2. No trailing commas
3. All strings must be properly escaped
4. All fields must be present
5. Each question MUST have exactly 4 options (A, B, C, D)
6. Difficulty must be one of: Beginner, Intermediate, Advanced
7. Points must be a number with value 2 (without quotes) and must always be followed by "points" :
8. Make sure all quotes are properly closed
9. Each option must have both 'id' and 'text' fields
11. Every property strictly followed by a colon
12. Numbers should not be in quotes
13. No line breaks within string values
14. Start with {{ and end with }}
15. Strictly use commas between all properties
16. No extra text before or after the JSON
17. Strictly avoid using the slash or backslash in the output in any form.
    Niether in question nor in answer or any other text of output.

Example format:
{{
    "quizTitle": "Basic Math Quiz",
    "totalQuestions": 3,
    "timeLimit": 10,
    "questions": [
        {{
            "id": "q1",
            "question": "What is 2 + 2?",
            "options": [
                {{"id": "A", "text": "3"}},
                {{"id": "B", "text": "4"}},
                {{"id": "C", "text": "5"}},
                {{"id": "D", "text": "6"}}
            ],
            "correctOptionId": "B",
            "points": 2
        }}
    ]
}}
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1500
            # response_format="json"
        )
        
        if not chat_completion or not chat_completion.choices:
            raise ValueError("Empty response from Groq API")
            
        response_content = chat_completion.choices[0].message.content
        
        if not response_content or response_content.strip() == "":
            raise ValueError("Empty response content from Groq API")
        
        return response_content
        
    except Exception as e:
        raise ValueError(f"Failed to generate quiz content: {str(e)}")

def take_user_input_and_create_test(topic):
    
    client = initialize_groq_client(api_key)
    if not client:
        raise ValueError("Failed to initialize Groq client")

    try:
        quiz_content = quiz_creator_model(client, topic)

        if not quiz_content:
            raise ValueError("No quiz content generated")

        import json           
        return json.loads(quiz_content)

    except Exception as e:
        raise ValueError(f"Error generating quiz: {str(e)}")
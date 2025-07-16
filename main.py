# Load environment variables first
from dotenv import load_dotenv
import os
load_dotenv()

# Gemini setup & prompt design 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tools import tools
from schema import MoodResponse

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
parser = PydanticOutputParser(pydantic_object=MoodResponse)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a caring mental health check-in assistant.

    When a user shares how they're feeling, you MUST:
    1. First, call the suggest_activity tool with their mood to get a personalized activity recommendation
    2. Then, call the log_mood_entry tool to save their mood entry
    3. Finally, respond with empathy and share the activity suggestion

    Always use the available tools to help the user. Don't try to guess what the tools would return.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
])

# Chat interface setup
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Create agent and executor
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Chat loop
chat_history = []
print("Welcome to the Mental Health Check-in Bot (type 'exit' to quit)")

while True:
    q = input("\nYou: ")
    if q.lower() == "exit":
        print("Take care! Remember to check in with yourself regularly. 💙")
        break
    
    try:
        # Execute the agent with user input
        response = executor.invoke({
            "query": q,
            "chat_history": chat_history
        })
        
        # Display the agent's response directly
        print(f"\nBot: {response['output']}")
        
        # Update chat history
        chat_history.extend([
            HumanMessage(content=q),
            AIMessage(content=response['output'])
        ])

    except Exception as e:
        print(f"Sorry, I encountered an error: {e}")
        print("Let's try again...") 

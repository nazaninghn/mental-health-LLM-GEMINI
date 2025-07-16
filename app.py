from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
load_dotenv()

# Import your existing bot components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import tools

app = Flask(__name__)

# Initialize your bot
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

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

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Store chat history (in production, use a database)
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    try:
        # Get bot response
        response = executor.invoke({
            "query": user_message,
            "chat_history": chat_history
        })
        
        bot_response = response['output']
        
        # Update chat history
        chat_history.extend([
            HumanMessage(content=user_message),
            AIMessage(content=bot_response)
        ])
        
        return jsonify({
            'success': True,
            'response': bot_response
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
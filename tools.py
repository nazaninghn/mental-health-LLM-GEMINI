import os,datetime
from langchain.tools import Tool
#log file is about usesr 
def log_mood_entry(entry:str):
    os.makedirs("logs",exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"logs/{datetime.datetime.now():%Y%m%d}_log.txt"
    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"[{timestamp}]{entry}\n")
    return f"Entry logged at {timestamp}"
def suggest_activity(mood:str):    # suggest self care sctivity based on emotional
    mood = mood.lower()
    if "anxious" in mood:
        return "Try a 5-minute deep breathing exercise or write down what you're feeling"
    elif "sad" in mood:
        return "Consider calling a friend or going for a walk in nature"
    elif "angry" in mood:
        return "Pause and do a quick grounding exercise-focus on your breath for 60 seconds"
    elif "tired" in mood:
        return "Take a power nap or rest your eyes for 10 minutes."
    elif "calm" in mood or "happy" in mood:
        return "That's great! Maybe write in your gratitude journal or listen to your favorite song."
    else:
        return "Take a few minutes for yourself - perhaps light stretching or mindful breathing"
#register the tools
tools = [
    Tool(name="log_mood_entry",func=log_mood_entry,description="Log the  user's mood check-in to a file"),
    Tool(name="suggest_activity",func= suggest_activity,description="Return an activity recommendation based on mood")
]
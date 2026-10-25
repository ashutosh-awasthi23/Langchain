import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import wikipedia
def setup_environment():
    load_dotenv()  # .env file se environment variables load karta hai
    wikipedia.set_user_agent("Langchain_Agent_PoC/1.0")  # Ye nayi line add karein
def get_langfuse_handler():
    """Langfuse tracker initialize karta hai observability ke liye."""
    return CallbackHandler()

def initialize_llm():
    """Core LLM engine setup karta hai."""
    return ChatOpenAI(temperature=0, model="gpt-4o-mini")

def get_tools(llm):
    """External sources (jaise Wikipedia) se data laane ke liye tools load karta hai."""
    return load_tools(["wikipedia"], llm=llm)

def setup_memory():
    """Pichli chat history yaad rakhne ke liye memory setup karta hai."""
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def create_research_agent(llm, tools, memory):
    """LLM, Tools, aur Memory ko jod kar Agent create karta hai."""
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

def main():
    # 1. Environment aur Tracker Setup Karein
    setup_environment()
    langfuse_handler = get_langfuse_handler()
    
    # 2. Components Initialize Karein
    llm = initialize_llm()
    tools = get_tools(llm)
    memory = setup_memory()

    
    # 3. Agent Banayein
    agent = create_research_agent(llm, tools, memory)
    
    # 4. Agent ko Test Karein
    print("Round 1: Asking a question...")
    response1 = agent.run(
        "Hi, mera naam Rahul hai. MS Dhoni ka janm kahan hua tha?", 
        callbacks=[langfuse_handler]
    )
    print("Answer 1:", response1)

    print("\nRound 2: Testing Memory...")
    response2 = agent.run(
        "Mera naam kya hai aur maine abhi kiske baare me pucha tha?", 
        callbacks=[langfuse_handler]
    )
    print("Answer 2:", response2)

if __name__ == "__main__":
    main()
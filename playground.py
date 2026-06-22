import os
from dotenv import load_dotenv

load_dotenv()

from agno.os import AgentOS
from agents import coordinator_agent, retriever_agent, general_agent
from workflow import workflow

agent_os = AgentOS(
    agents=[retriever_agent, general_agent],
    teams=[coordinator_agent],
    workflows=[workflow],
)
app = agent_os.get_app()

if __name__ == "__main__":
    print("\n🚀 Starting Agno Playground UI Server on Local Engine...")
    print("Connecting your local custom DuckDB database context to the interface.")
    agent_os.serve(app="playground:app", reload=True, port=7777)
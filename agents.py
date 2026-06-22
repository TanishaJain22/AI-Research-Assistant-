import logging
import os
from dotenv import load_dotenv

load_dotenv()

from agno.agent import Agent
from agno.team import Team, TeamMode
from agno.models.openai import OpenAIChat

from knowledge_base import get_knowledge_base
from tools import web_search_tool, calculator_tool

logger = logging.getLogger("assistant.agents")

knowledge_base = get_knowledge_base()

# Used by evaluation.py
last_retrieved_docs = []
last_selected_agent = None


def search_local_documents(query: str) -> str:
    """
    Search the local vector database.
    """

    global last_retrieved_docs
    global last_selected_agent

    last_selected_agent = "Retriever Agent"

    logger.info(
        f"Retriever Agent searching knowledge base for: {query}"
    )

    results = knowledge_base.vector_db.search(
        query,
        limit=2
    )

    last_retrieved_docs = results

    if not results:
        return "No relevant documents found."

    output = []

    for idx, doc in enumerate(results, start=1):
        score = doc.meta_data.get("score", 0.0)

        output.append(
            f"""
            Chunk {idx}
            Document: {doc.name}
            Similarity Score: {score:.4f}

            Content:
            {doc.content}
            """
                    )

    return "\n\n".join(output)


RETRIEVER_AGENT_INSTRUCTIONS = """
You are the Retriever Agent.

You answer questions using information retrieved from the local knowledge base.

Rules:
1. Always use the search_local_documents tool.
2. Use only retrieved content when answering.
3. Do not use outside knowledge.
4. If information is not found in retrieved content, only state "no information found", and give no extra and unrequired information.
5. Summarize retrieved information into a concise answer.
"""

 
retriever_agent = Agent(
    name="Retriever Agent",
    model=OpenAIChat(
      id="openai/gpt-oss-120b:free",
      api_key=os.getenv("OPENAI_API_KEY"),
      base_url="https://openrouter.ai/api/v1"),
    tools=[search_local_documents],
    instructions=RETRIEVER_AGENT_INSTRUCTIONS,
    markdown=True,
)


GENERAL_AGENT_INSTRUCTIONS = """
You are the General Agent.

You handle:
- General reasoning
- Mathematics
- Current information
- Web searches

Rules:
1. Use Calculator Tool whenever computation is required.
2. Use Web Search Tool whenever external or current information is required.
3. Do not claim a tool was used unless it was actually called.
4. If no tool is required, answer normally.
"""

general_agent = Agent(
    name="General Agent",
    model=OpenAIChat(
      id="openai/gpt-oss-120b:free",
      api_key=os.getenv("OPENAI_API_KEY"),
      base_url="https://openrouter.ai/api/v1"),
    tools=[
        web_search_tool,
        calculator_tool
    ],
    instructions=GENERAL_AGENT_INSTRUCTIONS,
    markdown=True,
)


COORDINATOR_AGENT_INSTRUCTIONS = """
You are the Coordinator Agent.

Your responsibility is to determine which specialist should handle the user's request.

Available Specialists:

1. Retriever Agent

   * Answers questions using information stored in the local knowledge base.

2. General Agent

   * Handles reasoning, calculations, web search, current events, and general world knowledge.

Knowledge Base Overview:

The local knowledge base contains reference material related to:

* Authentic Neapolitan pizza preparation and standards
* Pizza dough fermentation and baking processes
* Ingredient sourcing requirements
* Traditional Edomae sushi preparation
* Sushi rice (shari) preparation and service standards
* Fish aging techniques and premium tuna handling
* Japanese culinary knife craftsmanship
* Culinary techniques involving fermentation, temperature control, and traditional food preparation

Routing Principles:

* Route to the Retriever Agent whenever the user's question appears likely to be answered using the information contained in the knowledge base.
* Route to the General Agent when the question requires reasoning, calculations, web search, current information, or knowledge outside the scope of the knowledge base.
* When uncertain, prefer the Retriever Agent if the query appears related to the domains represented in the knowledge base.

Rules:

1. Never answer the user's question yourself.
2. Always delegate to exactly one specialist.
"""

coordinator_agent = Team(
    name="Coordinator Agent",
    model=OpenAIChat(
      id="openai/gpt-oss-120b:free",
      api_key=os.getenv("OPENAI_API_KEY"),
      base_url="https://openrouter.ai/api/v1"),
    members=[
        retriever_agent,
        general_agent
    ],
    mode=TeamMode.route,
    instructions=COORDINATOR_AGENT_INSTRUCTIONS,
    # debug_mode=True,
    markdown=True,
)

import os
import logging

from dotenv import load_dotenv

load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            "logs/agent_activity.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("assistant.main")


def handle_query(query: str):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    import agents
    import re

    agents.last_retrieved_docs = []

    logger.info(
        f"Received Query: {query}"
    )

    response = agents.coordinator_agent.run(query)

    for member_response in response.member_responses:
        print(member_response.agent_name)  # which agent was selected
        print(member_response.content)

    if agents.last_retrieved_docs:

        print("\nRetrieved Chunks:")

        for idx, doc in enumerate(
            agents.last_retrieved_docs,
            start=1
        ):

            score = doc.meta_data.get(
                "score",
                0.0
            )

            snippet = (
                doc.content[:150]
                .replace("\n", " ")
            )

            print(
                f"\n[{idx}] {doc.name}"
            )

            print(
                f"Score: {score:.4f}"
            )

            print(
                f"Snippet: {snippet}..."
            )

    agents.last_retrieved_docs = []


if __name__ == "__main__":

    print(
        "\nAI Research Assistant"
    )

    while True:

        query = input(
            "\nEnter Query (exit to quit): "
        ).strip()

        if query.lower() in (
            "exit",
            "quit",
            "q"
        ):
            break

        if not query:
            continue

        handle_query(query)
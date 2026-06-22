import logging
import re

from rich.console import Console
from rich.table import Table

import agents
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("assistant.evaluate")
console = Console()


# TEST_CASES = [
#     {
#         "query": "How warm should sushi rice be before serving?",
#         "expected_agent": "Retriever Agent",
#         "expected_docs": ["japanese_sushi_rules.txt"],
#     },
#     {
#         "query": "How long should premium bluefin tuna be aged?",
#         "expected_agent": "Retriever Agent",
#         "expected_docs": ["japanese_sushi_rules.txt"],
#     },
#     {
#         "query": "Which knife uses a single-bevel edge?",
#         "expected_agent": "Retriever Agent",
#         "expected_docs": [
#             "japanese_sushi_rules.txt",
#             "culinary_techniques_overview.md",
#         ],
#     },
#     {
#         "query": "What ingredients are required in authentic pizza dough?",
#         "expected_agent": "Retriever Agent",
#         "expected_docs": ["italian_pizza_secrets.txt"],
#     },
#     {
#         "query": "Why is pizza dough fermented for a long time?",
#         "expected_agent": "Retriever Agent",
#         "expected_docs": [
#             "italian_pizza_secrets.txt",
#             "culinary_techniques_overview.md",
#         ],
#     },
#     {
#         "query": "Calculate 17% of 245",
#         "expected_agent": "General Agent",
#         "expected_docs": [],
#     },
#     {
#         "query": "Search latest news about Agno framework",
#         "expected_agent": "General Agent",
#         "expected_docs": [],
#     },
#     {
#         "query": "Explain supervised learning",
#         "expected_agent": "General Agent",
#         "expected_docs": [],
#     },
# ]

TEST_CASES = [
    {
        "query": "At what temperature should sushi rice be served to customers?",
        "expected_agent": "Retriever Agent",
        "expected_docs": ["japanese_sushi_rules.txt"],
    },
    {
        "query": "How long should premium tuna mature before use?",
        "expected_agent": "Retriever Agent",
        "expected_docs": ["japanese_sushi_rules.txt"],
    },
    {
        "query": "What knife is designed to slice fish in one continuous stroke?",
        "expected_agent": "Retriever Agent",
        "expected_docs": [
            "japanese_sushi_rules.txt",
            "culinary_techniques_overview.md",
        ],
    },
    {
        "query": "What ingredients are mandatory in traditional Naples-style pizza dough?",
        "expected_agent": "Retriever Agent",
        "expected_docs": ["italian_pizza_secrets.txt"],
    },
    {
        "query": "Why is pizza dough left to rest for many hours?",
        "expected_agent": "Retriever Agent",
        "expected_docs": [
            "italian_pizza_secrets.txt",
            "culinary_techniques_overview.md",
        ],
    },
    {
        "query": "Calculate 17% of 245",
        "expected_agent": "General Agent",
        "expected_docs": [],
    },
    {
        "query": "Find recent news about multi-agent AI frameworks",
        "expected_agent": "General Agent",
        "expected_docs": [],
    },
    {
        "query": "Explain supervised learning to a beginner",
        "expected_agent": "General Agent",
        "expected_docs": [],
    },
]

# def extract_routing_metadata(response_text: str):

#     agent = "UNKNOWN"
#     reason = "N/A"

#     agent_match = re.search(
#         r"ROUTING_DECISION:\s*(Retriever Agent|General Agent)",
#         response_text,
#         re.IGNORECASE,
#     )

#     if agent_match:
#         value = agent_match.group(1).strip()

#         if value.lower().startswith("retriever"):
#             agent = "Retriever Agent"
#         else:
#             agent = "General Agent"

#     reason_match = re.search(
#         r"REASON:\s*(.*?)\s*(?:DELEGATED_RESPONSE:|$)",
#         response_text,
#         re.IGNORECASE | re.DOTALL,
#     )

#     if reason_match:
#         reason = " ".join(
#             reason_match.group(1).split()
#         )

#     return agent, reason


def evaluate_retrieval(
    expected_docs: list[str],
    retrieved_docs: list[str],
) -> bool:

    if not expected_docs:
        return True

    expected = {d.lower() for d in expected_docs}
    retrieved = {d.lower() for d in retrieved_docs}

    return len(expected.intersection(retrieved)) > 0


def run_evaluation():

    table = Table(
        title="Semantic Routing + Retrieval Evaluation"
    )

    table.add_column(
        "Query",
        width=40,
    )

    table.add_column(
        "Expected Agent",
        width=18,
    )

    table.add_column(
        "Actual Agent",
        width=18,
    )

    table.add_column(
        "Routing",
        width=8,
    )

    table.add_column(
        "Expected Docs",
        width=30,
    )

    table.add_column(
        "Retrieved Docs",
        width=30,
    )

    table.add_column(
        "Retrieval",
        width=10,
    )

    table.add_column(
        "Top Score",
        width=10,
    )

    routing_pass_count = 0
    retrieval_pass_count = 0

    for test in TEST_CASES:

        query = test["query"]

        expected_agent = test["expected_agent"]

        expected_docs = test["expected_docs"]

        logger.info(
            f"Evaluating query: {query}"
        )

        response = agents.coordinator_agent.run(query)

        # response_text = getattr(
        #     response,
        #     "content",
        #     str(response)
        # )

        actual_agent = "UNKNOWN"
        for member_response in response.member_responses:
            actual_agent = member_response.agent_name


        # actual_agent, routing_reason = (
        #     extract_routing_metadata(
        #         response_text
        #     )
        # )

        routing_pass = (
            actual_agent == expected_agent
        )

        if routing_pass:
            routing_pass_count += 1

        retrieved_docs = []
        scores = []

        for doc in agents.last_retrieved_docs:

            retrieved_docs.append(
                doc.name
            )

            scores.append(
                doc.meta_data.get(
                    "score",
                    0.0,
                )
            )

        retrieval_pass = evaluate_retrieval(
            expected_docs,
            retrieved_docs,
        )

        if retrieval_pass:
            retrieval_pass_count += 1

        top_score = (
            f"{max(scores):.4f}"
            if scores
            else "-"
        )

        table.add_row(
            query,
            expected_agent,
            actual_agent,
            "✅" if routing_pass else "❌",
            # routing_reason,
            (
                ", ".join(expected_docs)
                if expected_docs
                else "N/A"
            ),
            (
                ", ".join(retrieved_docs)
                if retrieved_docs
                else "N/A"
            ),
            "✅" if retrieval_pass else "❌",
            top_score,
        )

        agents.last_retrieved_docs = []

    console.print(table)

    console.print(
        f"\n[bold green]Routing Accuracy:[/bold green] "
        f"{routing_pass_count}/{len(TEST_CASES)}"
    )

    console.print(
        f"[bold blue]Retrieval Accuracy:[/bold blue] "
        f"{retrieval_pass_count}/{len(TEST_CASES)}"
    )


if __name__ == "__main__":
    run_evaluation()
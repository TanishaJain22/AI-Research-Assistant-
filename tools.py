from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.calculator import CalculatorTools

# Define external tools
web_search_tool = DuckDuckGoTools(fixed_max_results=5)
calculator_tool = CalculatorTools()

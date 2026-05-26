from tavily import TavilyClient
from dotenv import load_dotenv
import os
load_dotenv(override=True)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# response = tavily_client.search("Who is Leo Messi?")

# print(response)

response = tavily_client.search(
    query='"John Smith" CEO Acme Corp',
    exact_match=True
)
print(response)



context = tavily_client.get_search_context(query="What happened during the Burning Man floods?")

print(context)
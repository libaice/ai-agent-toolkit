import voyageai
import os
from dotenv import load_dotenv

load_dotenv(override=True)

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

result = vo.embed(["hello world"], model="voyage-4-large")

print(result.embeddings)


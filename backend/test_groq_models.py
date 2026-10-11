"""List available Groq models."""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

models = client.models.list()
for m in models.data:
    print(f"ID: {m.id}")
    print(f"  Name: {m.name}")
    print(f"  Context window: {getattr(m, 'context_window', 'N/A')}")
    print(f"  Owned by: {getattr(m, 'owned_by', 'N/A')}")
    print(f"  Created: {getattr(m, 'created', 'N/A')}")
    print()

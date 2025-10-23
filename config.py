import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://223.130.153.136:9201")
ELASTIC_USER = os.getenv("ELASTIC_USER", "elastic")
ELASTIC_PASS = os.getenv("ELASTIC_PASS")
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://223.130.153.136:9201")
ELASTIC_USER = os.getenv("ELASTIC_USER", "elastic")
ELASTIC_PASS = os.getenv("ELASTIC_PASS")

HELLAW_DB_HOST = os.getenv("HELLAW_DB_HOST", "localhost")
HELLAW_DB_USER = os.getenv("HELLAW_DB_USER", "root")
HELLAW_DB_PASSWORD = os.getenv("HELLAW_DB_PASSWORD", "")
HELLAW_DB_NAME = os.getenv("HELLAW_DB_NAME", "hellaw")

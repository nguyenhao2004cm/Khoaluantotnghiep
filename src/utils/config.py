import os
from dotenv import load_dotenv

load_dotenv()

ALPHA_KEY = os.getenv("ALPHA_KEY")
NEWS_KEY = os.getenv("NEWS_KEY")
FRED_KEY = os.getenv("FRED_KEY")

import os
from dotenv import load_dotenv, dotenv_values

# Path to your .env file
env_path = "../.env"

# Step 1: Clear existing env vars that are in the .env file
env_vars = dotenv_values(env_path)
for key in env_vars:
    os.environ.pop(key, None)

# Step 2: Reload from .env
load_dotenv(dotenv_path=env_path, override=True)

# Optional: print to confirm
print(os.environ.get("DEBUG_MODE"))  # Replace with a real key

from utils.config import Config

# Try to load a config file
config = Config.from_yaml("examples/llm_config.yaml")
print("Successfully imported Config from utils.config!")

import os
import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self, config_path: str = None):
        self.config_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.config_path)))
        config_path = os.path.join(base_dir, "config", "config.yml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def app_name(self) -> str:
        return self.config["app"]["name"]

    @property
    def app_version(self) -> str:
        return self.config["app"]["version"]

    @property
    def app_description(self) -> str:
        return self.config["app"]["description"]

    @property
    def vector_index_path(self) -> str:
        return self.config["data"]["vector_index_path"]

    @property
    def chunk_size(self) -> int:
        return int(self.config["data"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return int(self.config["data"]["chunk_overlap"])

    @property
    def batch_size(self) -> int:
        return int(self.config["data"].get("batch_size", 50))

    @property
    def embedding_provider(self) -> str:
        return self.config["models"]["embedding_provider"]

    @property
    def embedding_model_name(self) -> str:
        return self.config["models"]["embedding_model_name"]

    @property
    def llm_provider(self) -> str:
        return self.config["models"]["llm_provider"]

    @property
    def llm_model_name(self) -> str:
        return self.config["models"]["llm_model_name"]

    @property
    def k_retrieval(self) -> int:
        return int(self.config["models"]["k_retrieval"])

    @property
    def max_input_tokens(self) -> int:
        return int(self.config["models"]["max_input_tokens"])

    @property
    def max_output_tokens(self) -> int:
        return int(self.config["models"]["max_output_tokens"])

    @property
    def thinking_level(self) -> str:
        return self.config["models"]["thinking_level"]

    @property
    def input_price_per_million(self) -> float:
        return float(self.config["pricing"]["input_price_per_million"])

    @property
    def output_price_per_million(self) -> float:
        return float(self.config["pricing"]["output_price_per_million"])

    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")

        if not key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set in .env file!"
            )

        return key


settings = Config()
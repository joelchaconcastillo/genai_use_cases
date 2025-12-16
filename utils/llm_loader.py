import os
from typing import Any, Dict, List

import numpy as np
try:
    from huggingface_hub import InferenceClient  # type: ignore
except Exception:
    InferenceClient = None


class HFInferenceEmbeddings:
    """Wrapper to use Hugging Face Inference API as embeddings provider.

    Note: Requires `huggingface_hub` and `numpy` installed. This class uses the
    synchronous `InferenceClient.feature_extraction` endpoint to get embeddings.
    """

    def __init__(self, model_name: str, hf_token: str, batch_size: int = 8):
        if InferenceClient is None:
            raise RuntimeError("huggingface_hub.InferenceClient is required for HFInferenceEmbeddings")
        self.client = InferenceClient(api_key=hf_token)
        self.model_name = model_name
        self.batch_size = batch_size

    def _embed_text(self, text: str):
        res = self.client.feature_extraction(text, model=self.model_name)
        emb = np.array(res, dtype=float)
        if emb.ndim > 1:
            emb = emb[0]
        return emb

    def embed_documents(self, texts: List[str]):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str):
        return self._embed_text(text)


class LLM_Client:
    def __init__(self, model_config: Dict[str, Any]):
        if model_config.get("provider") == "hf":
            self.llm = self._init_hf_model(model_config)
        elif model_config.get("provider") == "gemini":
            self.llm = self._init_google_model(model_config)
        else:
            raise ValueError(f"Unsupported provider: {model_config.get('provider')}")

    def _init_hf_model(self, model_config: Dict[str, Any]):
        # Resolve model name and API key (api_key_env should have been loaded by manager)
        model_name = model_config.get("model")
        api_key_env = model_config.get("api_key_env")
        hf_token = None
        if api_key_env:
            hf_token = os.getenv(api_key_env)
        # instantiate the HF Inference embeddings wrapper
        return HFInferenceEmbeddings(model_name=model_name, hf_token=hf_token)

    def _init_google_model(self, model_config: Dict[str, Any]):
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        return ChatGoogleGenerativeAI(model=model_config.get("model"), temperature=model_config.get("temperature", 0.7))

class LLM_Manager:
    def __init__(self):
        import yaml
        with open("configs/llm_profiles.yaml", "r") as file:
            self.profiles = yaml.safe_load(file)["profiles"]

        for profile_name, config in self.profiles.items():
            print(f"Initialized LLM Client for profile: {profile_name} with config: {config}")
            #set env variable for api key
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv(config.get("api_key_env", ""))
            if api_key:
                os.environ[config["api_key_env"]] = api_key

            client = LLM_Client(config)

            test = client.llm.invoke("Hello, world!")
            print(test.content)


if __name__ == "__main__":
    llm = LLM_Manager()

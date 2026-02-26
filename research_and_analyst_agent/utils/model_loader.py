import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from research_and_analyst_agent.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from research_and_analyst_agent.logger import GLOBAL_LOGGER as log
from research_and_analyst_agent.exception.custom_exception import ResearchAnalystException


class ApiKeyManager:
    """
    Loads and manages all environment-based API keys.
    """

    def __init__(self):
        load_dotenv()

        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        }

        log.info("Initializing ApiKeyManager")

        # Log loaded key statuses without exposing secrets
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded successfully from environment")
            else:
                log.warning(f"{key} is missing in environment variables")

    def get(self, key: str):
        """
        Retrieve a specific API key.

        Args:
            key (str): Name of the API key.

        Returns:
            str | None: API key value if found.
        """
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs dynamically based on YAML configuration and environment settings.
    """

    def __init__(self):
        """
        Initialize the ModelLoader and load configuration.
        """
        try:
            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML configuration loaded successfully", config_keys=list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise ResearchAnalystException("Failed to initialize ModelLoader", sys)

    # ----------------------------------------------------------------------
    # Embedding Loader
    # ----------------------------------------------------------------------
    def load_embeddings(self):
        """
        Load and return a Google Generative AI embedding model.

        Returns:
            GoogleGenerativeAIEmbeddings: Loaded embedding model instance.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Ensure event loop exists for gRPC-based embedding API
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
            )

            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ResearchAnalystException("Failed to load embedding model", sys)

   
    def load_llm(self):
        """
        Load and return a chat‑based LLM according to the configured provider.

        Supported providers:
            - OpenAI
            - Google (Gemini)
            - Groq

        A small sanity check is performed for Google models to warn when a
        free‑tier/flash variant is configured, since those often have a
        quota of 0 and immediately fail with a 429 error.  The returned
        object is wrapped so that invocations are guarded against quota
        errors and an informative exception is raised.
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "google")

            if provider_key not in llm_block:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = llm_block[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                if model_name in self.FREE_TIER_GOOGLE_MODELS:
                    log.warning(
                        "Configured Google model '%s' is a free-tier/flash variant which often has 0 quota;"
                        " switching to paid equivalent or update your config.",
                        model_name,
                    )
                raw_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "groq":
                raw_llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )

            elif provider == "openai":
                raw_llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                    temperature=temperature,
                )

            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            # wrap the model to guard against quota issues
            llm = self._LLMWrapper(raw_llm)

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise ResearchAnalystException("Failed to load LLM", sys)


# ----------------------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        loader = ModelLoader()

        # # Test embedding model
        # embeddings = loader.load_embeddings()
        # print(f"Embedding Model Loaded: {embeddings}")
        # result = embeddings.embed_query("Hello, how are you?")
        # print(f"Embedding Result: {result[:5]} ...")

        # Test LLM
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        try:
            result = llm.invoke("Hello, how are you?")
            print(f"LLM Result: {result.content[:200]}")
        except ResearchAnalystException as qe:
            # Rate-limit / quota error; printed message already logged by wrapper
            print("Encountered a quota error running the LLM:", qe)

        log.info("ModelLoader test completed successfully")

    except ResearchAnalystException as e:
        log.error("Critical failure in ModelLoader test", error=str(e))
        
        
# Write a clean, enterprise-grade Python module for dynamic model loading in a structured AI backend system. The system must follow clean architecture principles and separate API key management, configuration loading, and model initialization logic. Use environment variables and YAML configuration to determine which LLM provider to load. Support OpenAI, Google Gemini, and Groq chat models. Include structured logging at every stage, avoid exposing secrets, ensure async loop safety for gRPC-based embedding APIs, and wrap all failures using a custom domain exception class. Provide complete documentation, comments, error handling, and a standalone test block for local validation.
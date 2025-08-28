from openai import OpenAI
from openai import (
    RateLimitError,
    InternalServerError,
)
import re
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
import time
import logging
import random
from . import configs
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed,
)

def retry_on_exception(
    func, exception_filter_func, multiplier=1, max_wait=40, max_attempts=100
):
    """Executes a function with retry logic for certain exceptions. Never retries on KeyboardInterrupt.
    Args:
        func: The function to execute with retries
        exception_filter_func: Function that checks if an exception needs to be retried
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function call
    """
    retry_function = retry(
        retry=(
            retry_if_not_exception_type(KeyboardInterrupt)
            & retry_if_exception(exception_filter_func)
        ),
        wait=wait_random_exponential(multiplier=multiplier, max=max_wait),
        stop=stop_after_attempt(max_attempts),
    )
    return retry_function(func)


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                logging.info(f"Attempting to call {func.__name__}")
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                if num_retries > max_retries:
                    logging.error(
                        f"Maximum number of retries ({max_retries}) exceeded for {func.__name__}."
                    )
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                logging.warning(
                    f"Retry #{num_retries} for {func.__name__} after encountering {e}. Waiting {delay} seconds before retrying..."
                )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                logging.exception(f"Unexpected exception during {func.__name__}: {e}")
                raise e

    return wrapper


class OpenAIAgent:

    def __init__(self):
        print("=== OpenAIAgent.__init__ called ===")
        if configs.model_name.startswith("trapi-"):
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    DefaultAzureCredential(),
                    ManagedIdentityCredential(),
                    AzureCliCredential(),
                ),
                scope,
            )
            if "gpt-4o" in configs.model_name.lower():    
                _model_name = 'gpt-4o'  # Ensure this is a valid model name
                model_version = '2024-11-20'  # Ensure this is a valid model version
                instance = 'msrne/shared' # Use the same instance as the working Azure configuration
                api_version = '2024-10-21' # Ensure this is a valid API version
            else:
                raise ValueError(f"Unsupported model name: {configs.model_name}. Only 'trapi-gpt-4o' is supported.")

            deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{_model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
            endpoint = f'https://trapi.research.microsoft.com/{instance}'

            print(f"FactScoreLite TRAPI Config:")
            print(f"  Model: {_model_name}")
            print(f"  Version: {model_version}")
            print(f"  Instance: {instance}")
            print(f"  Deployment: {deployment_name}")
            print(f"  Endpoint: {endpoint}")
            print(f"  API Version: {api_version}")

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential,
                api_version=api_version,
            )
            self.model_name = deployment_name  # Use the deployment name as the model name
            self.max_tokens = configs.max_tokens
            self.temp = configs.temp

        else:
            self.client = OpenAI()
            self.max_tokens = configs.max_tokens
            self.temp = configs.temp
            self.model_name = configs.model_name

    def generate(self, prompt):

        try:
            print(f"FactScoreLite API Call - Using model: {self.model_name}")
            response = retry_on_exception(
                self.client.chat.completions.create, self.need_to_be_retried
            )(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temp,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.exception(f"Error occurred while generating response: {e}")
            return "Failed to generate response."

    def need_to_be_retried(self, exception) -> bool:
        # List of fully qualified names of RateLimitError exceptions from various libraries
        _errors = [
            "openai.APIStatusError",
            "openai.InternalServerError",  # Add 503 errors
            "openai.APITimeoutError",
            "openai.error.Timeout",
            "openai.error.RateLimitError",
            "openai.error.ServiceUnavailableError",
            "openai.Timeout",
            "openai.APIError",
            "openai.APIConnectionError",
            "openai.RateLimitError",
            "openai.PermissionDeniedError",
            "openai.BadRequestError",
            # Add more as needed
        ]
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )

        need_to_retry = exception_full_name in _errors

        # Ignore error that are not rate limit errors
        if exception_full_name == "openai.APIStatusError":
            if not (
                "'status': 429" in exception.message  # Rate Limit Exceeded
                or "'status': 504" in exception.message  # Gateway Timeout
                or (  # A previous prompt was too large
                    "'status': 413" in exception.message
                    and "A previous prompt was too large." in exception.message
                )
            ):
                need_to_retry = False

        # Always retry InternalServerError (503) but with longer wait
        if exception_full_name == "openai.InternalServerError" and "'status': 503" in exception.message:
            need_to_retry = True

        return need_to_retry



# Ensure proper logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

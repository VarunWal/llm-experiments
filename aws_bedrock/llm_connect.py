"""Python Script to call an AWS Bedrock LLM (DeepSeek R1) using boto3"""

from dotenv import load_dotenv
import boto3
import json
import os
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

class BedrockDeepSeekClient:
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        region_name="us-east-1",
    ):
        """
        Initialize the Bedrock client for DeepSeek R1

        Args:
            aws_access_key_id: AWS Access Key ID (if None, will use environment variable)
            aws_secret_access_key: AWS Secret Access Key (if None, will use environment variable)
            region_name: AWS region where Bedrock is available
        """
        # Use provided credentials or fall back to environment variables
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.region_name = region_name

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials must be provided either as parameters or environment variables"
            )

        # Initialize the Bedrock runtime client
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        # DeepSeek R1 model ID in Bedrock
        # self.model_id = "deepseek.deepseek-r1-distill-llama-70b-v1:0"
        self.model_id = "us.deepseek.r1-v1:0"

    def call_deepseek_r1(self, prompt, max_tokens=1000, temperature=0.7, top_p=0.9):
        """
        Call DeepSeek R1 model via AWS Bedrock

        Args:
            prompt: The input prompt/message
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 to 1.0)
            top_p: Controls nucleus sampling (0.0 to 1.0)

        Returns:
            Generated response text
        """
        try:
            # Prepare the request body
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": [],
            }

            # Convert to JSON
            body_json = json.dumps(body)

            # Make the API call
            response = self.bedrock_client.invoke_model(
                body=body_json,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            # Parse the response
            response_body = json.loads(response.get("body").read())

            # Print the full response for debugging (remove this later)
            # print(f"Full response body: {response_body}")

            # DeepSeek R1 response format - try different possible keys
            generated_text = ""

            # Try common response keys
            if "generations" in response_body:
                # Format: {"generations": [{"text": "..."}]}
                generations = response_body["generations"]
                if generations and len(generations) > 0:
                    generated_text = generations[0].get("text", "")
            elif "completion" in response_body:
                # Format: {"completion": "..."}
                generated_text = response_body.get("completion", "")
            elif "choices" in response_body:
                # Format: {"choices": [{"text": "..."}]}
                choices = response_body["choices"]
                if choices and len(choices) > 0:
                    generated_text = choices[0].get("text", "")
            elif "output" in response_body:
                # Format: {"output": "..."}
                generated_text = response_body.get("output", "")
            elif "text" in response_body:
                # Format: {"text": "..."}
                generated_text = response_body.get("text", "")
            else:
                # If none of the above, print the keys to help debug
                print(
                    f"Unknown response format. Available keys: {list(response_body.keys())}"
                )
                return str(response_body)

            return generated_text.strip()

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            print(f"AWS Client Error ({error_code}): {error_message}")
            return None

        except BotoCoreError as e:
            print(f"BotoCore Error: {str(e)}")
            return None

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

    def chat_conversation(self, messages, max_tokens=1000, temperature=0.7, top_p=0.9):
        """
        Handle a conversation with multiple messages

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness
            top_p: Controls nucleus sampling

        Returns:
            Generated response text
        """
        # Format messages into a single prompt
        formatted_prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "user":
                formatted_prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            elif role == "system":
                formatted_prompt = f"System: {content}\n\n" + formatted_prompt

        formatted_prompt += "Assistant: "

        return self.call_deepseek_r1(formatted_prompt, max_tokens, temperature, top_p)


def main():
    """
    Example usage of the BedrockDeepSeekClient
    """
    # Option 1: Provide credentials directly
    # client = BedrockDeepSeekClient(
    #     aws_access_key_id="your_access_key_here",
    #     aws_secret_access_key="your_secret_key_here",
    #     region_name="us-east-1"
    # )

    # Option 2: Use environment variables (recommended)
    # Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment
    try:
        client = BedrockDeepSeekClient(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )

        # Simple prompt example
        print("=== Simple Prompt Example ===")
        prompt = "Explain quantum computing in simple terms."
        response = client.call_deepseek_r1(prompt, max_tokens=500)

        if response:
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
        else:
            print("Failed to get response")

        print("\n" + "=" * 50 + "\n")

        # Conversation example
        print("=== Conversation Example ===")
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What's the population of that city?"},
        ]

        response = client.chat_conversation(messages, max_tokens=300)

        if response:
            print("Conversation:")
            for msg in messages:
                print(f"{msg['role'].title()}: {msg['content']}")
            print(f"Assistant: {response}")
        else:
            print("Failed to get conversation response")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo use this script, either:")
        print(
            "1. Set environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
        print(
            "2. Or modify the script to pass credentials directly to BedrockDeepSeekClient()"
        )

    except Exception as e:
        print(f"Error initializing client: {e}")


if __name__ == "__main__":
    main()

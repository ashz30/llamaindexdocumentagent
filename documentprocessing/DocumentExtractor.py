from llama_index.llms.openai import OpenAI


class DocumentExtractor:
    def __init__(self, model="gpt-4"):
        """
        Initializes the DocumentExtractor with the GPT model.
        :param model: OpenAI GPT model (e.g., "gpt-4").
        """
        self.llm = OpenAI(model=model)

    def structure_data(self, document_text,classification, prompt_file_path):
        """
        Structures the key details from document text based on classification and prompt present in prompt file
        :param document_text: The document text to process.
        :param classification: The classification of the document (e.g., Tax Invoice, Bill of Supply).
        :param prompt_file_path: Path to the file containing prompts based on classification.
        :return: Extracted data in text format.
        """
        try:
            print("Tools call Extractor")

            # Read the prompt from the file based on classification
            prompt = self.get_prompt_from_file(classification, prompt_file_path)
            if not prompt:
                raise ValueError(f"No prompt found for classification: {classification}")

            # Construct the full prompt with the document text and dynamic prompt
            full_prompt = f"{prompt}\n\n{document_text}"

            # Extract data using GPT
            extracted_data = self.get_extracted_data_from_gpt(full_prompt)

            return extracted_data.text

        except Exception as e:
            print(f"Error during extraction: {e}")
            return {"error": str(e)}

    def get_prompt_from_file(self, classification, prompt_file_path):
        """
        Retrieves the appropriate prompt from a file based on classification.
        :param classification: The classification (e.g., Tax Invoice, Bill of Supply).
        :param prompt_file_path: Path to the prompt file.
        :return: The prompt for the classification.
        """
        try:
            with open(prompt_file_path, "r") as file:
                prompts = file.readlines()

            # Search for the prompt corresponding to the classification
            for prompt in prompts:
                if prompt.startswith(f"{classification}:"):
                    return prompt[len(classification)+2:].strip()  # Skip the "classification: " part

            # If no prompt is found, return None
            return None
        except FileNotFoundError:
            print(f"Prompt file {prompt_file_path} not found.")
            return None

    def get_extracted_data_from_gpt(self, prompt):
        """
        Sends the constructed prompt to GPT for extraction.
        :param prompt: The prompt containing document text and extraction instructions.
        :return: The extracted data as a JSON string.
        """
        try:
            response = self.llm.complete(prompt)
            return response # Clean up the response
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            return '{"error": "Extraction failed"}'

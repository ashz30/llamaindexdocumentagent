
from llama_index.llms.openai import OpenAI

# Set your OpenAI API key here


class DocumentClassifier:
    def __init__(self, model="gpt-4"):
        self.llm = OpenAI(model=model)  # Using OpenAI from LlamaIndex


    def classify_documents(self, document_text, prompt):
        """Executes the prompt to classify the document_text on topics present in the prompt, outputs the classification value"""

        try:
            print("Tools call Classifier")

            # Create the prompt by adding the document content
            final_prompt = f"{prompt}:\n\n{document_text[:200]}"

            # Get classification from GPT-4
            classification = self.get_classification_from_gpt(final_prompt)

            print(f"Text: {document_text[:200]}...")  # Print first 200 chars for preview
            print(f"Classification: {classification}")
            print("-" * 80)
            topic = classification.text.split(',')[0].strip()
            return topic

        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def get_classification_from_gpt(self, prompt):
        """
        Sends the prompt to GPT-4 for classification and retrieves the response.
        :param prompt: The prompt to send to GPT for classification.
        :return: The classification result from GPT.
        """
        try:
            # Use LlamaIndex's OpenAI integration to get a response
            response = self.llm.complete(prompt)
            return response # Clean up the response
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            return "Error in classification"

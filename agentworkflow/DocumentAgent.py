from llama_index.core.agent.workflow import FunctionAgent
from documentprocessing.DocumentClassifier import DocumentClassifier
from documentprocessing.DocumentExtractor import DocumentExtractor
from documentprocessing.DocumentReader import DocumentReader


from llama_index.llms.openai import OpenAI
from llama_index.core.tools import  FunctionTool

import nest_asyncio

nest_asyncio.apply()

reader = DocumentReader()
extractor = DocumentExtractor()
classifier = DocumentClassifier()

reader_tool = FunctionTool.from_defaults(
    reader.read_documents
)

classifier_tool = FunctionTool.from_defaults(
    classifier.classify_documents
)

extractor_tool = FunctionTool.from_defaults(
    extractor.structure_data
)
tools = [reader_tool, classifier_tool, extractor_tool]

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing operations.",
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are an agent that can perform operations using tools."
)


async def main():
    classification_categories = "Amazon Bill, flipkart bill, GST bill, Financial results for Blue Prism"

    response = await workflow.run(user_msg="perform these actions sequentially - 1. read document in folder '../docs' , Step 2. Classify the text and choose 1 category from these : " + classification_categories + " and lastly Step 3. Structure the key details from the prompt present in prompt file '../documentprocessing/prompts.txt'")

    print(response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import asyncio

    asyncio.run(main())




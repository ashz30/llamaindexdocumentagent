from documentprocessing.DocumentClassifier import DocumentClassifier
from documentprocessing.DocumentExtractor import DocumentExtractor
from documentprocessing.DocumentReader import DocumentReader


if __name__ == "__main__":
    input_dir = "./docs"
    reader = DocumentReader()
    document_text = reader.read_documents(input_dir=input_dir)


    # Step 1: Classify the document
    classification_categories = "Amazon Bill, flipkart bill, GST bill, Financial results for Blue Prism"

    classifier = DocumentClassifier()
    classification = classifier.classify_documents(document_text = document_text, prompt = "Classify the document's content into one of the topics with a confidence value. Output should be in the format <Classification>, <Confidence>, Topics available for classificaiton are - " + classification_categories)

    # Step 2: Extract data based on classification result
    extractor = DocumentExtractor()
    data_extracted = extractor.extractnStructure_data(document_text=document_text, classification=classification )
    print(data_extracted)


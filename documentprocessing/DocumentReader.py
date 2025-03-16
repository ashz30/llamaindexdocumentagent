from llama_index.core import SimpleDirectoryReader

class DocumentReader:

    def read_documents(self,input_dir: str) -> str:
        """reads different types of documents in the give directory in input_dir and outputs the unstructured text"""
        try:
            print("Tools call Reader")
            reader = SimpleDirectoryReader(input_dir=input_dir)
            documents = reader.load_data()
            document_text = ""
            if not documents:
                raise ValueError(f"No documents found in directory: {input_dir}")

            # Iterate over the documents and print their filename and text
            for document in documents:
                filename = document.metadata.get("file_name", "Unknown Filename")
                text = document.text if hasattr(document, 'text') else "No text found in document."
                #print(f"Filename: {filename}")
                #print(f"Text: {text}")
                document_text = document_text + text
            return document_text
        except FileNotFoundError as e:
            print(f"Error: The directory {input_dir} was not found. {e}")
        except AttributeError as e:
            print(f"Error: Missing expected attribute in document. {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    reader = DocumentReader("1")
    reader.read_documents(input_dir="../docs")
import PyPDF2
import pg_vector_util
from langchain.text_splitter import CharacterTextSplitter


def extract_text(filepath):
    # Open the PDF file in read-binary mode
    with open(filepath, "rb") as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Create an empty string to store the text
        text = ""

        # Loop through each page in the PDF file
        for page_num in range(len(pdf_reader.pages)):
            # Update the progress bar
            print("progress: " + str(page_num + 1) + "/" + str(len(pdf_reader.pages)))

            # Get the page object
            page_obj = pdf_reader.pages[page_num]

            # Extract the text from the page
            page_text = page_obj.extract_text()

            # Add the text to the string
            text += page_text

    return text


def split_text(text):
    text_splitter = CharacterTextSplitter()
    return text_splitter.split_text(text)


def write_textstr_to_db(text):
    pg_vector_util.pg_conn()
    split_text(text)


# if __name__ == "__main__":
#     text = extract_text("image/frax.pdf")
#     print(text)

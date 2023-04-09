import PyPDF2


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


# if __name__ == "__main__":
#     text = extract_text("image/frax.pdf")
#     print(text)

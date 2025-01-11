import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader

# Initialize the ChatGroq model
grok_key = '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'
llm = ChatGroq(temperature=0, api_key=grok_key)

# Function to extract text from PDF
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Define summarization prompt
summarize_template = """Write a concise summary of the following content in {words_limit} words. 
At the end, generate the most important points in bullet points as highlights.

{text}

Summary:

Important Highlights:
- 
- 
- 
"""


prompt = PromptTemplate(input_variables=["text", "words_limit"], template=summarize_template)

# Function to split text into manageable chunks
def split_text(text, chunk_size=4000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to summarize chunks and combine the results
def summarize_large_text(text, words_limit, chunk_size=4000, chunk_overlap=500):
    # Split text into chunks
    chunks = split_text(text, chunk_size, chunk_overlap)
    st.write(f"Divided text into {len(chunks)} chunks.")

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {i + 1}...")
        chain = LLMChain(llm=llm, prompt=prompt)
        chunk_summary = chain.run({
            "text": chunk,
            "words_limit": words_limit // len(chunks)  # Proportionate limit per chunk
        })
        chunk_summaries.append(chunk_summary)

    # Combine chunk summaries into a single text
    combined_text = " ".join(chunk_summaries)

    # Generate a final summary of the combined text
    st.write("Generating final summary...")
    final_chain = LLMChain(llm=llm, prompt=prompt)
    final_summary = final_chain.run({
        "text": combined_text,
        "words_limit": words_limit
    })

    return final_summary

# Streamlit interface
st.title("Document Summarizer")
st.write("Upload a document (PDF or text), or provide a URL to summarize content.")
words_limit = st.number_input("Maximum words in summary", min_value=50, max_value=500, value=200, step=50)

# Option to upload a file or provide a URL
option = st.radio("Select Input Method", ("Upload File", "Enter URL", "Enter Text"))

if option == "Upload File":
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        # Extract text from the uploaded document
        if uploaded_file.type == "application/pdf":
            st.write("Extracting text from PDF...")
            text = extract_pdf_text(uploaded_file)
        elif uploaded_file.type == "text/plain":
            st.write("Reading text file...")
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format!")
            st.stop()

        # Summarize text
        st.write("Generating summary...")
        summary = summarize_large_text(text=text, words_limit=words_limit)

        # Display summary
        st.subheader("Summary")
        st.write(summary)
        
        # Add download button for summary
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

elif option == "Enter URL":
    url = st.text_input("Enter the URL of a webpage")
    if url:
        try:
            st.write("Loading content from URL...")
            loader = WebBaseLoader(url)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])

            # Summarize text
            st.write("Generating summary...")
            summary = summarize_large_text(text=text, words_limit=words_limit)

            # Display summary
            st.subheader("Summary")
            st.write(summary)
            
            # Add download button for summary
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"An error occurred while loading the URL: {e}")

elif option == "Enter Text":
    text_input = st.text_area("Enter your text here", height=300)
    if st.button("Generate Summary"):
        if text_input:
            # Summarize text
            st.write("Generating summary...")
            summary = summarize_large_text(text=text_input, words_limit=words_limit)

            # Display summary
            st.subheader("Summary")
            st.write(summary)
            
            # Add download button for summary
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

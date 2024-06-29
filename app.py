import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import streamlit as st

# Set your Hugging Face API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def get_pdf(pdfs):
  """
  This function extracts text from a list of PDF files and combines them.

  Args:
      pdfs: A list of uploaded PDF files.

  Returns:
      A string containing the combined text extracted from all PDFs.
  """
  text = ""
  for pdf in pdfs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text


def get_chunks(text):
  """
  This function splits the extracted text into smaller chunks for processing.

  Args:
      text: The combined text extracted from all PDFs.

  Returns:
      A list of text chunks.
  """
  splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  chunks = splitter.split_text(text)
  return chunks


def get_vectors(chunks):
  """
  This function generates embeddings (numerical representations) for each text chunk.

  Args:
      chunks: A list of text chunks.

  Effects:
      Saves the generated embeddings to a local FAISS index named "faiss_index".
  """
  embedding = HuggingFaceBgeEmbeddings(
      model_name="BAAI/bge-small-en-v1.5",
      encode_kwargs={'normalize_embeddings': True}
  )
  vector_store = FAISS.from_texts(chunks, embedding)
  vector_store.save_local("faiss_index")


def conversational_chain():
  """
  This function defines the conversational chain using a large language model (LLM).

  Returns:
      A Langchain question-answering chain object.
  """
  model = HuggingFaceEndpoint(
      repo_id="mistralai/Mistral-7B-Instruct-v0.3",
      model_kwargs={"max_length": 500},
      temperature=0.1,
      timeout=400
  )

  # Access chat history from Streamlit session state
  chat_history = st.session_state.get("chat_history", [])

  # Build the prompt template incorporating previous context (chat history)
  prompt = """Answer the question as detailed as possible considering the following context:

  Previous Context: """
  for entry in chat_history:
    prompt += f"{entry['question']}: {entry['answer']}\n"

  prompt += """

  Current Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
  """

  template = PromptTemplate(template=prompt, input_variables=['context', 'question'])
  chain = load_qa_chain(llm=model, chain_type='stuff', prompt=template)
  return chain


def user_input(user_question):
  """
  This function handles user input, retrieves relevant documents, and generates a response.

  Args:
      user_question: The question asked by the user.

  Effects:
      Updates the chat history in Streamlit session state.
  """
  embedding = HuggingFaceBgeEmbeddings(
      model_name="BAAI/bge-small-en-v1.5",
      encode_kwargs={'normalize_embeddings': True}
  )
  vector_store = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
  docs = vector_store.similarity_search(user_question, k=2)

  chain = conversational_chain()
  response = chain({
      'input_documents': docs,
      "question": user_question
  }, return_only_outputs=True)
  print(response)
  st.write("Reply: ", response["output_text"])
  if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
  st.session_state["chat_history"].append({"question": user_question, "answer": response["output_text"]})

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vectors(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

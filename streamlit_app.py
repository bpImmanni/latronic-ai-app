# app.py

import streamlit as st
import pandas as pd

from langchain.document_loaders import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import tempfile
import os
import smtplib
from email.message import EmailMessage

# --- Page Settings ---
st.set_page_config(page_title="RAG File Processor", layout="wide")
st.title("📁 AI Document Processor + Email Sender")
st.markdown("Upload a CSV or Excel file, ask a question, and get a new file emailed to you.")

# --- Inputs ---
api_key = st.text_input("🔑 OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📂 Upload file (CSV or XLSX)", type=["csv", "xlsx"])
prompt = st.text_area("💬 What would you like me to do with this data?")
email = st.text_input("📧 Your email (must be @latronicsolutions.com)")
process_button = st.button("🚀 Process and Email Me")

# --- Validate Email Domain ---
def valid_company_email(email):
    return email.endswith("@latronicsolutions.com")

# --- Process File with LangChain ---
def process_file(file, file_type, prompt, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if file_type == "csv":
        loader = CSVLoader(file_path=tmp_path)
    else:
        df = pd.read_excel(tmp_path)
        df.to_csv(tmp_path + ".csv", index=False)
        loader = CSVLoader(file_path=tmp_path + ".csv")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.from_documents(chunks, embeddings)
    llm = OpenAI(openai_api_key=api_key)

    prompt_template = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow up Input: {question}
    Standalone question:
    """)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        condense_question_prompt=prompt_template,
        return_source_documents=False,
        verbose=False
    )

    result = qa_chain({"question": prompt, "chat_history": []})
    answer = result['answer']

    # Save output
    output_file = f"processed_output.csv"
    pd.DataFrame([{"Prompt": prompt, "Answer": answer}]).to_csv(output_file, index=False)
    return output_file, answer

# --- Email Output File ---
def send_email(recipient_email, file_path):
    msg = EmailMessage()
    msg["Subject"] = "📎 Your Processed File"
    msg["From"] = "bhanuprakash6841@gmail.com"
    msg["To"] = recipient_email
    msg.set_content("Attached is the file you requested.")

    with open(file_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(file_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("bhanuprakash6841@gmail.com", "Immanni@2204")
        smtp.send_message(msg)

# --- Main App Logic ---
if process_button:
    if not uploaded_file:
        st.error("Please upload a file.")
    elif not prompt:
        st.error("Please enter your prompt.")
    elif not email:
        st.error("Please enter your email.")
    elif not email.endswith("@latronicsolutions.com"):
        st.error("Only @latronicsolutions.com emails are allowed.")
    elif not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        with st.spinner("Processing..."):
            try:
                file_type = uploaded_file.name.split(".")[-1]
                output_file, result = process_file(uploaded_file, file_type, prompt, api_key)

                send_email(email, output_file)
                st.success(f"✅ Email sent to {email} with result.")
                st.success(f"🧠 AI Result: {result}")

                with open(output_file, "rb") as f:
                    st.download_button("⬇️ Download Output", f.read(), file_name=output_file)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

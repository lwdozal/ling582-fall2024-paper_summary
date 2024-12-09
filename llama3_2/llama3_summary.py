
#global content

import torch
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm as notebook_tqdm

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr
print("All libraries imported successfully!")

hf_token = "your token here" #write

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "PY007/TinyLlama-1.1B-Chat-v0.1"
model_id = "PY007/TinyLlama-1.1B-step-50K-105b"
tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)
# model = AutoModel.from_pretrained(model_id, use_auth_token = hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token = hf_token)
print("downloaded tiny models")


# Save it locally (optional for reuse)
# model.save_pretrained("./models/TinyLlama-1.1B-step")
# tokenizer.save_pretrained("./models/tokenizerTinyLlama-1B-step")

#Read in the models you saved to use for future analysis
# tokenizer = AutoTokenizer.from_pretrained("./models/Llama-3.2-3B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("./models/Llama-3.2-3B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("./models/TinyLlama-1.1B-Chat-v0.1")
# model = AutoModelForCausalLM.from_pretrained("./models/tokenizerTinyLlama")
# Use a sentence transformer model to encode chunks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Set pad token to EOS token
tokenizer.pad_token_id = tokenizer.eos_token_id

###############################

# convert the pdf into machine readable text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

#Chunk the data: i.e. split the data into parts to get easier summaries
# note: the standard 'chunk size' is about 5 sentences, or a paragraph
def split_text_into_chunks(text, chunk_size=300, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


### Setup the context for the questions to support the model

# Get the chunk that is most relevant to the question
def find_most_relevant_chunk(question, chunks, chunk_embeddings):
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    most_relevant_idx = np.argmax(similarities)
    return chunks[most_relevant_idx]

#Look into top relevant chunks and combine them to provide better responses?
def find_top_relevant_chunks(question, chunks, chunk_embeddings, top_n=4):
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings).flatten()
    top_indices = np.argsort(similarities)[-top_n:]  # Get indices of top `n` chunks
    return " ".join([chunks[idx] for idx in top_indices])



## QA for the model

# Create a formatted prompt

## with semantic search
# generate a prompt for multiple questions with context
def create_prompt(question, relevant_chunk):
    ## relevant chunk can be the top-most chunk or the top-most n chunks
    ## you decide in the answer_question() function
    return f"\nContext:\n{relevant_chunk}\n\nQuestion:\n{question}\n\nAnswer:"


def ask_question(intro, prompt):
    # Format input as a prompt
    prompt = f"Introductory information: {intro}\n" + prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    # outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=1025, num_return_sequences=1)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=5000, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# provide relevant chunks for each question
def answer_question(intro, questions, chunks, chunk_embeddings):
    text_blob = ''
    for question in notebook_tqdm(questions, desc="Processing Questions", total=len(questions), dynamic_ncols=True):
        # Find the most relevant context. This only provides one chunk
        relevant_chunk = find_most_relevant_chunk(question, chunks, chunk_embeddings)
        
        # Find the most relevant context. This provides the top n chunks, here we'll say the top 3 paragraphs
        # relevant_chunks = find_top_relevant_chunks(question, chunks, chunk_embeddings, top_n=3)

        #create a prompt for each question
        prompt = create_prompt(question, relevant_chunk)
        # print(prompt)
        
        #use for the top most relevant chunks
        # prompt = create_prompt(question, relevant_chunks)

        #run the model to ask the question
        answer = ask_question(intro, prompt)
        print(f"Q: {question}\nA: {answer}\n")
        text_blob += f"Q: {question}\nA: {answer}\n"
    return text_blob


#######################################################
def main():
    '''
    ####################
    DOWNLOAD THE MODEL WITH YOUR CREDIENTIALS!!!!!
    ####################

    First make sure you can log into huggingface and download the models.
    usually you only get the fine-grained token, here you need the 'write' token.

    the model used here is model_id = "meta-llama/Llama-3.2-3B-Instruct"
    Download the model and the tokenizer and save it to a local file so you don't spend too much time waiting every run.
    It also helps with runtime errors. The code is:

    # hf_token = "your huggingface token here" #write

    # tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)
    # model = AutoModelForCausalLM.from_pretrained(model_id, token = hf_token)

    # Save it locally (optional for reuse)
    # model.save_pretrained("./models/Llama-3.2-3B-Instruct")
    # tokenizer.save_pretrained("./models/tokenizerLlama-3.2-3B-I")

    '''
    ######
    #run through the code

    #read in the file
    pdf_path = "2022.lrec-Koeva_MultilingualImageCorpus.pdf"
    # convert the pdf into machine readable text
    document_text = extract_text_from_pdf(pdf_path)
    print(document_text)
    # split the text into chunks to provide relevant context
    chunks = split_text_into_chunks(document_text)
    print(chunks[0])
    # create embeddings for the chunks
    chunk_embeddings = embedding_model.encode(chunks)

    # Define a list of questions
    questions = [
        "What is the main topic discussed in this document?",
        "What is this paper about?",
        "What are the authors proposing?",
        "What is the motivation for the work?",
        "What is the approach or innovation?",
        "What are the results and how do they compare with competing approaches?",
        "Is the comparison between the results and competing approaches fair?",
        "What are the takeaways according to the authors?",
        "What are the takeaways according to you?",
        "Would you use this?  If so, how/where would you use this?",
        "What problems remain and what are the next steps?"
    ]

    # create prompt intro if needed
    prompt1 = "As a student of Deep Learning and Natural Language Processing doing research in the field. Use this document to answer the following questions.\n\n"
    
    # run the model to chat with the LLM
    response = answer_question(prompt1, questions, chunks, chunk_embeddings)
    print(response)


    with open("./responses.txt", "w", encoding="utf-8") as file:
        file.write(response)

if __name__ == "__main__":
    main()

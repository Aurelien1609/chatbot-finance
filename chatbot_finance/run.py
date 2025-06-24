import gradio as gr
import chromadb
import os
from transformers import AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST")
CHROMA_DB_COLLECTION = "financial_reports"


llm = LlamaCpp(
    model_path="data/models/mistral-7b-instruct-v0.2.Q4_0.gguf",
    n_ctx=8192,
    n_batch=64,
    temperature=0.0,
    max_tokens=512,
    n_threads=6,
    verbose=False,
    use_mlock=True,
)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
vectorstore = Chroma(client=client, collection_name="financial_reports", embedding_function=embedding_model)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Contexte : {context}\n\nQuestion : {question}\nRéponse :"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Fonction pour tronquer les documents
def format_docs(docs, max_tokens=1500):
    # Concatène contenu des documents puis tronque tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    content = "\n\n".join(doc.page_content for doc in docs)
    tokens = tokenizer.encode(content)
    return tokenizer.decode(tokens[:max_tokens])

# Fonction pour extraire contexte via retriever
def get_context_with_sources(question: str):
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs)
    sources = [doc.metadata.get("filename", "Source inconnue") for doc in docs]
    return context, sources

# Fonction principale chat sans répéter le contexte dans l’historique
def chat(user_input, chat_history):
    # 1. Extraire contexte + sources
    context, sources = get_context_with_sources(user_input)

    # 2. Construire historique sans contexte
    previous_messages = "\n".join([f"Utilisateur : {q}\nAssistant : {a}" for q, a in chat_history])

    # 3. Créer prompt complet pour le modèle
    prompt_input = f"{previous_messages}\nUtilisateur : {user_input}"
    final_prompt = prompt_template.format(context=context, question=prompt_input)

    # 4. Générer la réponse
    response = llm(final_prompt)

    # 5. Ajouter les sources à la réponse (format simple)
    sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    response_with_sources = response + sources_text

    # 6. Mettre à jour l'historique (sans contexte)
    chat_history.append((user_input, response_with_sources))
    return chat_history, chat_history

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Chatbot avec Mistral + Contexte ChromaDB")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Votre question")
    clear = gr.Button("Effacer la conversation")
    history = gr.State([])

    msg.submit(chat, inputs=[msg, history], outputs=[chatbot, history])
    clear.click(lambda: ([], []), outputs=[chatbot, history])

demo.launch()
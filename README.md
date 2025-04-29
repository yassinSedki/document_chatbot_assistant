# document_chatbot_assistant

The Document Chatbot Assistant is a Streamlit-based application that allows users to upload PDF documents and interact with them through a chatbot interface. The chatbot uses advanced language models and embeddings to analyze the content of the uploaded documents and provide concise, context-aware answers to user queries.
![image](https://github.com/user-attachments/assets/fd01ea06-b0f9-440a-bc08-b964c255e92f)

Features
  * Upload and process multiple PDF documents.
  * Automatically split documents into manageable text chunks for analysis.
  * Use FAISS for efficient document retrieval and search indexing.
  * Leverage Hugging Face models for embeddings and language generation.
  * Provide concise answers with references to document sources.
    
How It Works
Upload PDFs: Users can upload one or more PDF files through the sidebar.
Document Processing: The application processes the PDFs into text chunks and creates a searchable vector store using FAISS.
Chat Interaction: Users can ask questions about the uploaded documents, and the chatbot provides answers based on the document content.
Source References: The chatbot includes references to the relevant sections of the documents for transparency.

Requirements
  * Python 3.8 or higher
  * Hugging Face API token (stored in a .env file)
  * Required Python libraries (see requirements.txt)
   
Installation and Activation
    Clone the Repository
    
    #Set Up a Virtual Environment
      git clone https://github.com/username/document_chatbot_assistant.git
      cd document_chatbot_assistant
    #Set Up a Virtual Environment
      python -m venv venv
      venv\Scripts\activate
    #Install Dependencies
      pip install -r requirements.txt
    #Set Up Environment Variables
      HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
    #Run the Application
      streamlit run chatbot.py
  



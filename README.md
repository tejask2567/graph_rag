# Graph-Based Retrieval-Augmented Generation (RAG) System

This repository contains a comprehensive graph-based RAG (Retrieval-Augmented Generation) system tailored for cybersecurity/penetration testing use-cases. The project leverages MongoDB and Neo4j as vector databases and uses the GROQ API with the Llama 3.1 model for data processing and retrieval.

## Features
- **Data Ingestion**: A Jupyter Notebook (`data_ingestion.ipynb`) for ingesting data into MongoDB and Neo4j.
- **Graph-Based Retrieval**: Integration with Langgraph and Langchain for efficient and context-aware data retrieval.
- **OCR Capability**: EasyOCR integration to extract text from user-uploaded images.
- **Streamlit Interface**: A user-friendly `app.py` to facilitate interactive data querying and visualization.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/graph-rag-app.git
   cd graph-rag-app
Install dependencies: Ensure you have Python 3.8 or above installed. Run:
 K
pip install -r requirements.txt
Set up the environment variables: The following environment variables need to be configured:

p K
os.environ["GROQ_API_KEY"] = "your-groq-api-key"
os.environ["NEO4J_URI"] = "your-neo4j-uri"
os.environ["NEO4J_USERNAME"] = "your-neo4j-username"
os.environ["NEO4J_PASSWORD"] = "your-neo4j-password"
also add your mongodb uri 
You can set these in a .env file or export them in your shell.

Run the data ingestion notebook: Open and run data_ingestion.ipynb to ingest your data into MongoDB and Neo4j.

Start the Streamlit app:
 K
streamlit run app.py
Usage
Data Ingestion: The data_ingestion.ipynb file contains the necessary code to process and load your data into the MongoDB and Neo4j databases.
App Interface: app.py provides a user-friendly interface for data retrieval. Users can upload images to extract text using EasyOCR, which will appear in the input bar for further querying.
Tech Stack
Backend: Python, FastAPI, Neo4j, MongoDB
Front-end: Streamlit
AI Model: Llama 3.1 (via GROQ API)
OCR: EasyOCR
Libraries: Langgraph, Langchain, PyMongo, Neo4j Python Driver
Future Enhancements
Integration of advanced analytics dashboards.
Enhanced user authentication and role-based access control.
Additional graph-based analytics and visualization features.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contributions
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Developed by Tejas K

Ensure to adjust the repository link, username, and any additional details as needed for your specific project.


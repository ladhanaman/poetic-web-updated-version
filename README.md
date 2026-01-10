# Poetic Camera

## Description

The Poetic Camera is a Streamlit web application that transforms images into poetry. Using a Retrieval-Augmented Generation (RAG) pipeline, it analyzes an uploaded or captured image, identifies key themes and objects, and generates a poem in the style of a selected poet.

## Key Features

- **Image-to-Poetry Generation**: Converts images into unique poems.
- **Poet Personas**: Choose from different poet styles (e.g., Emily Dickinson, Percy Bysshe Shelley, Walt Whitman).
- **RAG Pipeline**:
    - **Vision Analysis**: Uses a vision model to analyze the image and generate a descriptive narrative.
    - **Memory Retrieval**: Retrieves relevant poems from a vector database (Pinecone).
    - **Poem Generation**: Generates a new poem based on the image narrative and retrieved poems.
- **Audio Synthesis**: Converts the generated poem into speech.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/poetic-camera.git
    cd poetic-camera
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables**:
    Create a `.env` file in the root directory and add the following environment variables:

    ```env
    GROQ_API_KEY="your-groq-api-key"
    PINECONE_API_KEY="your-pinecone-api-key"
    GEMINI_API_KEY="your-gemini-api-key"
    COHERE_API_KEY="your-cohere-api-key"
    ```

    You can also set the following optional environment variables to use different models:

    ```env
    VISION_MODEL_ID="your-vision-model-id"
    GENERATOR_MODEL_ID="your-generator-model-id"
    ```

## How to Run the Application

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Use the application**:
    - Select a poet persona from the sidebar.
    - Choose an input method (upload an image or use your camera).
    - Click the "Generate poem with voice" button to generate the poem.

## Project Structure

```
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── scripts/                # Python scripts for the RAG pipeline
│   ├── architect.py        # RAG architect for selecting the best candidates
│   ├── audio.py            # Audio synthesis engine
│   ├── generator.py        # Poem generation
│   ├── retriever.py        # Poem retrieval from Pinecone
│   └── vision_client.py    # Vision model client
├── data/                   # Data files for the vector database
└── .env.example            # Example environment file
```

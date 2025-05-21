# The Tony Blair Arxiv

This is a simple LLM chat application that allows users ask question about Arxiv papers and journals. 

## Technology Stack
- **Backend**: Flask
- **Frontend**: Flask Templates, AlpineJS, TailwindCSS, HTMX
- **Database**: SQLite, Upstash Vector DB
- **LLM**: OpenAI GPT-3.5 Turbo

## How does it work?
1. Before everything, I run the script `vectorizer.py` to vectorize the papers and store them in the Upstash Vector DB using OpenAI embedding API. This allows the datasets to be searched and queried using vector similarity search. For this take home test purpose, I used only the abstract of the dataset  due to time constraints.
2. The user can ask a question about the paper using the chat interface.
3. The question that is sent to the backend will first be vectorized using OpenAI embedding API and then searched in the Upstash Vector DB to find the most similar papers.
4. The most similar papers are then retrieved and passed to the OpenAI GPT-3.5 Turbo model along with the instructions and questions to generate a response.
5. The question and LLM response then stored in the SQLite database for historical reference.
6. The response is then sent back to the frontend and displayed in the chat interface.

## How to run the project locally
1. This project use `uv` as the package manager. You can install it by referring to this [link](https://docs.astral.sh/uv/guides/install-python).
2. Then run `uv run main.py` to start the server.
3. Open your browser and go to `http://localhost:8080` to see the application.
4. For the first time, you need to input OpenAI API key and Upstash Token in the input field and click on the button to save it. You can use my OpenAI API key and Upstash Token that is sent to the submission email for testing purpose.
5. After that, you can start asking questions about the papers and journals.

## How to run the project in Docker
1. Make sure you have Docker installed on your machine.
2. Build the Docker image using the following command:
   ```bash
   docker build -t tony-blair-arxiv .
   ```
3. Run the Docker container using the following command:
   ```bash
   docker run -p 8080:8080 tony-blair-arxiv
   ```
4. Open your browser and go to `http://localhost:8080` to see the application.

## How to run tests using Docker
1. Run following command to run the tests:
   ```bash
   docker run tony-blair-arxiv python run_tests.py
   ```
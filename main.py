from flask import Flask, render_template, request, redirect, url_for, make_response
from upstash_vector import Index, Vector
from openai import OpenAI
from upstash_vector import Index, Vector

from models import db, History, HistoryMessage

# =====================================================
# Global instances
# =====================================================
openai_client = None  # Will be initialized in main()
upstash_index = None  # Will be initialized in main()

# =====================================================
# Set up OpenAI
# =====================================================
MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_REQUEST = 8191

# =====================================================
# Configuration
# =====================================================
app = Flask(__name__)

# SQLite configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["OPENAI_API_KEY"] = None
app.config["UPSTASH_TOKEN"] = None

db.init_app(app)

# =====================================================
# Initialization
# =====================================================
with app.app_context():
    db.create_all()


# =====================================================
# Helper functions
# =====================================================
def embed_texts(texts):
    response = openai_client.embeddings.create(model=MODEL, input=texts)
    return [e.embedding for e in response.data]


# =====================================================
# Routes
# =====================================================
@app.route("/api/v1/history/<int:history_id>")
def load_history(history_id):
    history = History.query.get_or_404(history_id)
    messages = HistoryMessage.query.filter_by(history_id=history_id).all()
    return render_template(
        "components/history.html", messages=messages, history_id=history_id
    )


@app.route("/api/v1/send-message", methods=["POST"])
def send_message():
    message_text = request.form.get("message", "")
    session_id_str = request.form.get("session_id")

    new_session_created = False
    history = None

    if session_id_str and session_id_str != "null" and session_id_str.strip() != "":
        try:
            current_session_id = int(session_id_str)
            history = History.query.get(current_session_id)
            if history:
                session_id = history.id
            else:  # Invalid session_id_str, create new session
                history = History(
                    title=message_text[:30] if message_text else "New Chat"
                )
                db.session.add(history)
                db.session.commit()
                session_id = history.id
                new_session_created = True
        except ValueError:  # session_id_str is not a valid int
            history = History(title=message_text[:30] if message_text else "New Chat")
            db.session.add(history)
            db.session.commit()
            session_id = history.id
            new_session_created = True
    else:  # No session_id_str, create new session
        history = History(title=message_text[:30] if message_text else "New Chat")
        db.session.add(history)
        db.session.commit()
        session_id = history.id
        new_session_created = True

    # Save user message
    user_msg_db = HistoryMessage(
        history_id=session_id, message=message_text, is_user=True
    )
    db.session.add(user_msg_db)
    db.session.commit()

    # Create a placeholder for bot message
    bot_msg_db = HistoryMessage(
        history_id=session_id, message="Thinking...", is_user=False, is_pending=True
    )
    db.session.add(bot_msg_db)
    db.session.commit()
    bot_message_id = bot_msg_db.id

    response_html = render_template(
        "components/thinking_message.html",
        user_message=message_text,
        bot_message_id=bot_message_id,
        session_id=session_id,
    )

    resp = make_response(response_html)
    if new_session_created:
        resp.headers["HX-Trigger"] = "newSessionCreated"
    return resp


@app.route("/api/v1/get-bot-reply/<int:bot_message_id>", methods=["GET"])
def get_bot_reply(bot_message_id):
    bot_msg_db_entry = HistoryMessage.query.get_or_404(bot_message_id)
    if not bot_msg_db_entry.is_pending:  # Check if already processed
        return render_template(
            "components/bot_reply_content.html", bot_message=bot_msg_db_entry.message
        )

    # Ensure openai_client is initialized - MOVED THIS CHECK TO THE BEGINNING
    if openai_client is None:
        # This case should ideally be handled by ensuring initialization at app start
        # or redirecting to initialization page if keys are missing.
        # For now, returning an error message.
        bot_msg_db_entry.message = "Error: OpenAI client not initialized."
        bot_msg_db_entry.is_pending = False
        db.session.commit()
        return render_template(
            "components/bot_reply_content.html", bot_message=bot_msg_db_entry.message
        )

    # Find the user message that prompted this bot reply.
    # This assumes the immediately preceding user message in the same session.
    user_msg_db_entry = (
        HistoryMessage.query.filter(
            HistoryMessage.history_id == bot_msg_db_entry.history_id,
            HistoryMessage.is_user == True,
            HistoryMessage.id
            < bot_msg_db_entry.id,  # User message must come before bot message
        )
        .order_by(HistoryMessage.id.desc())
        .first()
    )

    if not user_msg_db_entry:
        # Fallback or error: if no preceding user message found (should not happen in normal flow)
        # For simplicity, we'll use the bot_msg_db_entry's history_id to find any user message
        # This part might need more robust logic depending on exact requirements.
        # As a quick fix, let's assume the latest user message in the session.
        user_msg_db_entry = (
            HistoryMessage.query.filter_by(
                history_id=bot_msg_db_entry.history_id, is_user=True
            )
            .order_by(HistoryMessage.id.desc())
            .first()
        )
        if not user_msg_db_entry:
            # Still no user message, update bot message to error and return
            bot_msg_db_entry.message = "Error: Could not find user message context."
            bot_msg_db_entry.is_pending = False
            db.session.commit()
            return render_template(
                "components/bot_reply_content.html",
                bot_message=bot_msg_db_entry.message,
            )

    user_message_text = user_msg_db_entry.message

    # Step 0: Gather history messages
    history_messages = None
    if bot_msg_db_entry.history_id:
        # Fetch all messages in the session
        history_messages = (
            HistoryMessage.query.filter_by(history_id=user_msg_db_entry.history_id)
            .order_by(HistoryMessage.id.asc())
            .all()
        )

    # Step 1: Create vector embedding for the user message
    embedding = None
    if history_messages is None:
        # If no history messages, just use the user message
        embedding = embed_texts([user_message_text])[0]
    else:
        # If there are history messages, concatenate them with the user message
        history_texts = [msg.message for msg in history_messages if msg.is_user]
        combined_text = "\n".join(history_texts) + "\n" + user_message_text
        # Create vector embedding for the combined text
        embedding = embed_texts([combined_text])[0]

    # Step 2: Query top 10 most similar vectors
    knowledge = []
    results = upstash_index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True,
    )
    for result in results:
        if result.metadata and "abstract" in result.metadata:
            knowledge.append(result.metadata["abstract"])

    # Step 3: Ask OpenAI for a response
    knowledge_str = "\n".join(knowledge)
    instructions = (
        "You are a helpful assistant that helps people answer question about Arxiv papers and journals\n"
        "Use the following knowledge to answer the user's question.\n"
        f"Knowledge: {knowledge_str}\n\n\n"
        "RULES:\n"
        "1. Answer the user's question based on the knowledge provided.\n"
        "2. If the knowledge does not contain enough information, say 'I don't know'.\n"
        "3. If the knowledge contains enough information, answer the user's question.\n"
        "4. If the user asks a question that is not related to the provided knowledge, say 'I don't know'."
    )

    if history_messages is not None:
        instructions += "\n\n\n"
        instructions += "CHAT HISTORIES\n"
        # If there are history messages, include them in the instructions
        for message in history_messages:
            if message.is_user:
                instructions += f"\nUser: {message.message}\n"
            else:
                instructions += f"\nAssistant: {message.message}\n"

    try:
        response = openai_client.chat.completions.create(  # Assuming standard OpenAI client v1.x
            model="gpt-3.5-turbo",  # Ensure this model is appropriate
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_message_text},
            ],
        )
        bot_reply_text = response.choices[0].message.content.strip()
    except Exception as e:
        # Log the error e
        bot_reply_text = "Sorry, I encountered an error while generating a response."

    # Update the bot message in DB
    bot_msg_db_entry.message = bot_reply_text
    bot_msg_db_entry.is_pending = False
    db.session.commit()

    return render_template(
        "components/bot_reply_content.html", bot_message=bot_reply_text
    )


@app.route("/initialize", methods=["POST"])
def initialize():
    global openai_client
    global upstash_index

    app.config["OPENAI_API_KEY"] = request.form.get("openai_api_key")
    app.config["UPSTASH_TOKEN"] = request.form.get("upstash_token")

    # Initialize OpenAI client with user-provided API key
    openai_api_key = request.form.get("openai_api_key")
    openai_client = OpenAI(api_key=openai_api_key)

    # Initialize Upstash index with user-provided token
    upstash_token = request.form.get("upstash_token")
    upstash_index = Index(
        url="https://capable-midge-9649-eu1-vector.upstash.io",
        token=upstash_token,
    )

    return redirect(url_for("index"))


@app.route("/")
def index():
    if app.config["OPENAI_API_KEY"] is None or app.config["UPSTASH_TOKEN"] is None:
        return render_template("initialization.html")
    sessions = History.query.order_by(History.id.desc()).all()
    return render_template("index.html", sessions=sessions)


@app.route("/sidebar")
def get_sidebar():
    sessions = History.query.order_by(History.id.desc()).all()
    return render_template("components/sidebar.html", sessions=sessions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=True)

import pytest
from main import app, db, History, HistoryMessage # Import db and models
from unittest.mock import patch, MagicMock # For mocking
import main as main_module # To access main.py's global variables for assertions

# Fixture to configure the app for testing and manage database per test
@pytest.fixture(autouse=True) # autouse=True to apply to all tests
def app_context_setup():
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:", # Use in-memory SQLite for tests
        "OPENAI_API_KEY": None, 
        "UPSTASH_TOKEN": None,
        "SERVER_NAME": "localhost.test" # For url_for if used outside request context
    })

    with app.app_context():
        db.create_all()

    yield # Test runs here

    with app.app_context():
        db.session.remove() # Clear session
        db.drop_all() # Drop all tables

@pytest.fixture
def client(): 
    return app.test_client()

# Mock for the global main.openai_client instance
@pytest.fixture
def mock_main_openai_client():
    with patch('main.openai_client', new_callable=MagicMock) as mock_client_instance:
        mock_embeddings_create = MagicMock()
        embedding_obj = MagicMock()
        embedding_obj.embedding = [0.1, 0.2, 0.3]
        mock_embeddings_create.return_value.data = [embedding_obj]
        mock_client_instance.embeddings.create = mock_embeddings_create

        mock_chat_completions_create = MagicMock()
        message_obj = MagicMock()
        message_obj.content = "Mocked bot reply"
        choice_obj = MagicMock()
        choice_obj.message = message_obj
        mock_chat_completions_create.return_value.choices = [choice_obj]
        mock_client_instance.chat.completions.create = mock_chat_completions_create
        yield mock_client_instance

# Mock for the global main.upstash_index instance
@pytest.fixture
def mock_main_upstash_index():
    with patch('main.upstash_index', new_callable=MagicMock) as mock_index_instance:
        mock_query = MagicMock()
        result_obj1 = MagicMock(metadata={"abstract": "Mocked abstract 1"})
        result_obj2 = MagicMock(metadata={"abstract": "Mocked abstract 2"})
        mock_query.return_value = [result_obj1, result_obj2]
        mock_index_instance.query = mock_query
        yield mock_index_instance

def test_initialize_route(client):
    """Test the /initialize route. Mocks OpenAI and Index constructors."""
    with patch('main.OpenAI') as MockOpenAIConstructor, \
         patch('main.Index') as MockUpstashIndexConstructor:
        
        mock_openai_instance = MagicMock()
        MockOpenAIConstructor.return_value = mock_openai_instance
        
        mock_upstash_instance = MagicMock()
        MockUpstashIndexConstructor.return_value = mock_upstash_instance

        response = client.post('/initialize', data={
            'openai_api_key': 'test_openai_key',
            'upstash_token': 'test_upstash_token'
        })
        assert response.status_code == 302 
        assert response.location == '/' 

        MockOpenAIConstructor.assert_called_once_with(api_key='test_openai_key')
        MockUpstashIndexConstructor.assert_called_once_with(
            url="https://capable-midge-9649-eu1-vector.upstash.io",
            token='test_upstash_token'
        )
        
        assert app.config["OPENAI_API_KEY"] == 'test_openai_key'
        assert app.config["UPSTASH_TOKEN"] == 'test_upstash_token'
        
        # Assert that the global variables in main.py were updated
        assert main_module.openai_client is mock_openai_instance
        assert main_module.upstash_index is mock_upstash_instance

        response_index = client.get('/')
        assert response_index.status_code == 200
        assert b"ArXiv LLM" in response_index.data
        assert b"Initialize API Keys" not in response_index.data

def test_index_initialized(client):
    """Test the index route when API keys are set via /initialize."""
    # First, initialize the app by calling the /initialize endpoint
    with patch('main.OpenAI', return_value=MagicMock()), \
         patch('main.Index', return_value=MagicMock()):
        client.post('/initialize', data={
            'openai_api_key': 'test_openai_key',
            'upstash_token': 'test_upstash_token'
        })
    
    response = client.get('/')
    assert response.status_code == 200
    assert b"ArXiv LLM" in response.data
    assert b"Initialize API Keys" not in response.data

def test_send_message_new_session(client):
    """Test /api/v1/send-message creating a new session."""
    app.config["OPENAI_API_KEY"] = "fake_key" # Required for some internal logic if any
    app.config["UPSTASH_TOKEN"] = "fake_token"

    response = client.post('/api/v1/send-message', data={
        'message': 'Hello Tony!',
        'session_id': '' 
    })
    assert response.status_code == 200
    assert b"Thinking..." in response.data
    assert b"Hello Tony!" in response.data
    assert response.headers.get("HX-Trigger") == "newSessionCreated"

    with app.app_context():
        history = History.query.order_by(History.id.desc()).first()
        assert history is not None
        assert history.title == "Hello Tony!"[:30]
        
        user_msg = HistoryMessage.query.filter_by(history_id=history.id, is_user=True).first()
        assert user_msg is not None
        assert user_msg.message == "Hello Tony!"

        bot_msg = HistoryMessage.query.filter_by(history_id=history.id, is_user=False).first()
        assert bot_msg is not None
        assert bot_msg.message == "Thinking..."
        assert bot_msg.is_pending == True

def test_send_message_existing_session(client):
    """Test /api/v1/send-message with an existing session."""
    app.config["OPENAI_API_KEY"] = "fake_key"
    app.config["UPSTASH_TOKEN"] = "fake_token"

    with app.app_context():
        history = History(title="Existing Chat")
        db.session.add(history)
        db.session.commit()
        session_id = history.id

    response = client.post('/api/v1/send-message', data={
        'message': 'Another question',
        'session_id': str(session_id)
    })
    assert response.status_code == 200
    assert b"Thinking..." in response.data
    assert b"Another question" in response.data
    assert response.headers.get("HX-Trigger") is None

    with app.app_context():
        user_msgs = HistoryMessage.query.filter_by(history_id=session_id, is_user=True).all()
        assert len(user_msgs) == 1
        assert user_msgs[0].message == "Another question"

        bot_msgs = HistoryMessage.query.filter_by(history_id=session_id, is_user=False, is_pending=True).all()
        assert len(bot_msgs) == 1
        assert bot_msgs[0].message == "Thinking..."

def test_get_bot_reply_pending(client, mock_main_openai_client, mock_main_upstash_index):
    """Test /api/v1/get-bot-reply when the reply is pending."""
    app.config["OPENAI_API_KEY"] = "fake_key_for_test"
    app.config["UPSTASH_TOKEN"] = "fake_token_for_test"
    # mock_main_openai_client and mock_main_upstash_index fixtures are active

    with app.app_context():
        history = History(title="Test Chat")
        db.session.add(history)
        db.session.commit()
        
        user_msg = HistoryMessage(history_id=history.id, message="User question", is_user=True)
        db.session.add(user_msg)
        bot_msg = HistoryMessage(history_id=history.id, message="Thinking...", is_user=False, is_pending=True)
        db.session.add(bot_msg)
        db.session.commit()
        bot_message_id = bot_msg.id

    response = client.get(f'/api/v1/get-bot-reply/{bot_message_id}')

    assert response.status_code == 200
    assert b"Mocked bot reply" in response.data

    mock_main_openai_client.embeddings.create.assert_called_once()
    mock_main_upstash_index.query.assert_called_once()
    mock_main_openai_client.chat.completions.create.assert_called_once()

    with app.app_context():
        updated_bot_msg = HistoryMessage.query.get(bot_message_id)
        assert updated_bot_msg.message == "Mocked bot reply"
        assert updated_bot_msg.is_pending == False

def test_get_bot_reply_processed(client):
    """Test /api/v1/get-bot-reply when the reply is already processed."""
    app.config["OPENAI_API_KEY"] = "fake_key" 
    app.config["UPSTASH_TOKEN"] = "fake_token"

    with app.app_context():
        history = History(title="Test Chat Processed")
        db.session.add(history)
        db.session.commit()
        
        user_msg = HistoryMessage(history_id=history.id, message="User question", is_user=True)
        db.session.add(user_msg)
        bot_msg = HistoryMessage(history_id=history.id, message="Already processed reply", is_user=False, is_pending=False)
        db.session.add(bot_msg)
        db.session.commit()
        bot_message_id = bot_msg.id
    
    response = client.get(f'/api/v1/get-bot-reply/{bot_message_id}')
    assert response.status_code == 200
    assert b"Already processed reply" in response.data

def test_get_bot_reply_no_user_message_context(client, mock_main_openai_client, mock_main_upstash_index):
    """Test /api/v1/get-bot-reply when no user message context can be found."""
    app.config["OPENAI_API_KEY"] = "fake_key"
    app.config["UPSTASH_TOKEN"] = "fake_token"

    with app.app_context():
        history = History(title="No User Context Chat")
        db.session.add(history)
        db.session.commit()
        
        bot_msg = HistoryMessage(history_id=history.id, message="Thinking...", is_user=False, is_pending=True)
        db.session.add(bot_msg)
        db.session.commit()
        bot_message_id = bot_msg.id

    response = client.get(f'/api/v1/get-bot-reply/{bot_message_id}')

    assert response.status_code == 200
    assert b"Error: Could not find user message context." in response.data

    with app.app_context():
        updated_bot_msg = HistoryMessage.query.get(bot_message_id)
        assert updated_bot_msg.message == "Error: Could not find user message context."
        assert updated_bot_msg.is_pending == False
    
    mock_main_openai_client.embeddings.create.assert_not_called()
    mock_main_upstash_index.query.assert_not_called()
    mock_main_openai_client.chat.completions.create.assert_not_called()

def test_load_history(client):
    """Test /api/v1/history/<history_id>."""
    app.config["OPENAI_API_KEY"] = "fake_key" 
    app.config["UPSTASH_TOKEN"] = "fake_token"

    with app.app_context():
        history = History(title="Test History Load")
        db.session.add(history)
        db.session.commit()
        session_id = history.id
        
        msg1 = HistoryMessage(history_id=session_id, message="User says hi", is_user=True)
        msg2 = HistoryMessage(history_id=session_id, message="Bot says hello", is_user=False)
        db.session.add_all([msg1, msg2])
        db.session.commit()

    response = client.get(f'/api/v1/history/{session_id}')
    assert response.status_code == 200
    assert b"User says hi" in response.data
    assert b"Bot says hello" in response.data

def test_get_sidebar(client):
    """Test /sidebar."""
    app.config["OPENAI_API_KEY"] = "fake_key"
    app.config["UPSTASH_TOKEN"] = "fake_token"

    with app.app_context():
        history1 = History(title="Chat Alpha")
        history2 = History(title="Chat Beta")
        db.session.add_all([history1, history2])
        db.session.commit()

    response = client.get('/sidebar')
    assert response.status_code == 200
    assert b"Chat Alpha" in response.data
    assert b"Chat Beta" in response.data

def test_get_bot_reply_main_openai_client_is_none(client, mock_main_upstash_index):
    """Test get_bot_reply when main.openai_client is None (e.g. after failed initialization)."""
    app.config["OPENAI_API_KEY"] = "fake_key_for_config" # Config key is set
    app.config["UPSTASH_TOKEN"] = "fake_token_for_config" # Config key is set
    # main.openai_client itself is None for this test path

    with app.app_context():
        history = History(title="OpenAI Client None Test")
        db.session.add(history)
        db.session.commit()
        
        user_msg = HistoryMessage(history_id=history.id, message="User question", is_user=True)
        db.session.add(user_msg)
        bot_msg = HistoryMessage(history_id=history.id, message="Thinking...", is_user=False, is_pending=True)
        db.session.add(bot_msg)
        db.session.commit()
        bot_message_id = bot_msg.id

    # Patch main.openai_client to be None for this specific test's execution of the route
    with patch('main.openai_client', None):
        response = client.get(f'/api/v1/get-bot-reply/{bot_message_id}')

    assert response.status_code == 200
    assert b"Error: OpenAI client not initialized." in response.data # From initial check in get_bot_reply

    with app.app_context():
        updated_bot_msg = HistoryMessage.query.get(bot_message_id)
        assert updated_bot_msg.message == "Error: OpenAI client not initialized."
        assert updated_bot_msg.is_pending == False

# Add this section to make the file directly executable
if __name__ == "__main__":
    import sys
    try:
        import pytest
        # Run the tests
        exit_code = pytest.main(["-v", __file__])
        sys.exit(exit_code)
    except ImportError:
        print("Error: pytest is not installed. Please install it using 'pip install pytest'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


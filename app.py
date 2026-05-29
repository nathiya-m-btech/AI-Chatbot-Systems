
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image
from dotenv import load_dotenv

import datetime
import tempfile
import os

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# =========================================================
# GEMINI API CONFIGURATION
# =========================================================

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# SECURITY SETTINGS
# =========================================================

# Maximum upload size = 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi'}

# =========================================================
# FILE VALIDATION FUNCTION
# =========================================================

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# =========================================================
# CHAT HISTORY STORAGE
# =========================================================

chat_history = []

# =========================================================
# CHATBOT KNOWLEDGE
# =========================================================

knowledge = """
You are IntelliChat Pro, an advanced AI-powered chatbot.

Main Features:
- AI-powered chatbot
- Real-time responses
- NLP-based conversation
- Gemini AI integration
- Coding assistance
- Smart question answering
- Fast response generation
- Voice chatbot support
- Multi-language interaction
- Context-aware conversation
- AI summarization
- PDF chatbot support
- Sentiment analysis
- Professional AI assistance

Rules:
- Give short and clear answers
- Respond professionally
- Help with coding, AI, NLP, ML, and projects
- If question is unclear, ask for more details
- Maintain friendly communication
"""

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# FEATURE PAGES
# =========================================================

@app.route("/voicebot")
def voicebot():
    return render_template("voicebot.html")

# =========================================================
# IMAGE BOT
# =========================================================

@app.route("/imagebot", methods=["GET", "POST"])
def imagebot():

    if request.method == "GET":
        return render_template("imagebot.html")

    try:

        if "image" not in request.files:
            return jsonify({
                "response": "No image uploaded"
            })

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "response": "No image selected"
            })

        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                "response": "Invalid image format"
            })

        question = request.form.get(
            "question",
            "Describe this image"
        )

        image = Image.open(image_file.stream).convert("RGB")

        prompt = f"""
        You are an AI image understanding assistant.

        User Question:
        {question}

        Answer clearly based on the image.
        """

        response = model.generate_content([prompt, image])

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "response": "Image processing failed"
        })

# =========================================================
# VIDEO BOT
# =========================================================

@app.route("/videobot", methods=["GET", "POST"])
def videobot():

    if request.method == "GET":
        return render_template("videobot.html")

    try:

        if "video" not in request.files:
            return jsonify({
                "response": "No video uploaded"
            })

        video_file = request.files["video"]

        if video_file.filename == "":
            return jsonify({
                "response": "No video selected"
            })

        if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                "response": "Invalid video format"
            })

        question = request.form.get(
            "question",
            "Describe this video"
        )

        # Save temporarily
        temp_path = os.path.join(
            tempfile.gettempdir(),
            video_file.filename
        )

        video_file.save(temp_path)

        prompt = f"""
        You are a video assistant.

        NOTE:
        This system currently provides prompt-based
        video assistance only.

        User Question:
        {question}

        Give a helpful response.
        """

        response = model.generate_content(prompt)

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "response": "Video processing failed"
        })

# =========================================================
# PDF BOT
# =========================================================

@app.route("/pdfbot", methods=["GET", "POST"])
def pdfbot():

    if request.method == "GET":
        return render_template("pdfbot.html")

    try:

        if "pdf" not in request.files:
            return jsonify({
                "response": "No PDF uploaded"
            })

        pdf_file = request.files["pdf"]

        if pdf_file.filename == "":
            return jsonify({
                "response": "No PDF selected"
            })

        if not allowed_file(pdf_file.filename, ALLOWED_PDF_EXTENSIONS):
            return jsonify({
                "response": "Invalid PDF format"
            })

        question = request.form.get(
            "question",
            "Summarize this PDF"
        )

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

        if not text.strip():
            return jsonify({
                "response": "Could not extract text from PDF"
            })

        prompt = f"""
        You are an AI PDF assistant.

        PDF CONTENT:
        {text}

        USER QUESTION:
        {question}

        Give a clear and professional answer.
        """

        response = model.generate_content(prompt)

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "response": "PDF processing failed"
        })

# =========================================================
# SUMMARIZER PAGE
# =========================================================

@app.route("/summarizer")
def summarizer():
    return render_template("summarizer.html")

# =========================================================
# MULTILINGUAL PAGE
# =========================================================

@app.route("/multilingual")
def multilingual():
    return render_template("multilingual.html")

# =========================================================
# SENTIMENT PAGE
# =========================================================

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")

# =========================================================
# CODING BOT
# =========================================================

@app.route("/codingbot", methods=["GET", "POST"])
def codingbot():

    if request.method == "GET":
        return render_template("codingbot.html")

    try:

        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "response": "No message received"
            })

        user_message = data["message"]

        prompt = f"""
        You are an expert AI coding assistant.

        User Question:
        {user_message}

        Give:
        - Explanation
        - Code
        - Fix if needed
        """

        response = model.generate_content(prompt)

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "response": "Coding assistant failed"
        })

# =========================================================
# MAIN CHAT API
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "response": "No message received"
            })

        user_message = data["message"]

        current_time = datetime.datetime.now().strftime("%H:%M")

        # Store user message
        chat_history.append({
            "role": "user",
            "message": user_message,
            "time": current_time
        })

        prompt = f"""
        {knowledge}

        User Question:
        {user_message}
        """

        response = model.generate_content(prompt)

        bot_reply = response.text

        # Store bot response
        chat_history.append({
            "role": "bot",
            "message": bot_reply,
            "time": current_time
        })

        return jsonify({
            "response": bot_reply,
            "time": current_time
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "response": "Chatbot failed"
        })

# =========================================================
# TEXT SUMMARIZER API
# =========================================================

@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "summary": "No text received"
            })

        text = data["text"]

        prompt = f"""
        Summarize the following text clearly and shortly:

        {text}
        """

        response = model.generate_content(prompt)

        return jsonify({
            "summary": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "summary": "Summarization failed"
        })

# =========================================================
# SENTIMENT ANALYSIS API
# =========================================================

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():

    try:

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "sentiment": "No text received"
            })

        text = data["text"]

        prompt = f"""
        Analyze the sentiment of this text.

        Return only:
        Positive, Negative, or Neutral.

        Text:
        {text}
        """

        response = model.generate_content(prompt)

        return jsonify({
            "sentiment": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "sentiment": "Sentiment analysis failed"
        })

# =========================================================
# TRANSLATION API
# =========================================================

@app.route("/translate", methods=["POST"])
def translate():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "translated_text": "No data received"
            })

        text = data.get("text", "")
        language = data.get("language", "English")

        prompt = f"""
        Translate this text into {language}:

        {text}
        """

        response = model.generate_content(prompt)

        return jsonify({
            "translated_text": response.text
        })

    except Exception as e:

        app.logger.error(str(e))

        return jsonify({
            "translated_text": "Translation failed"
        })

# =========================================================
# GET CHAT HISTORY
# =========================================================

@app.route("/history", methods=["GET"])
def history():

    return jsonify(chat_history)

# =========================================================
# CLEAR CHAT HISTORY
# =========================================================

@app.route("/clear", methods=["POST"])
def clear_chat():

    global chat_history

    chat_history = []

    return jsonify({
        "message": "Chat history cleared"
    })

# =========================================================
# HEALTH CHECK ROUTE
# =========================================================

@app.route("/status")
def status():

    return jsonify({
        "status": "Online",
        "model": "gemini-2.5-flash",
        "version": "1.0"
    })

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )

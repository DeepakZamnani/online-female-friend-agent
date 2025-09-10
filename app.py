from flask import Flask, request, jsonify
from build_llm import build_llm  # Import your chain/llm object
from flask_cors import CORS
llm=build_llm()
app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Call your LLM chain
        response = llm.invoke({"input": user_message})

        # If response is a dict-like object
        if isinstance(response, dict) and "response" in response:
            output = response["response"]
        # If response is a LangChain Message object
        elif hasattr(response, "content"):
            output = response.content
        else:
            output = str(response)

        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

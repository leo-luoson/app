from flask import Flask, render_template


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        Page 1: Trade unit price prediction (static layout only).
        """
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        """
        Page 2: Trade feature dashboard (static layout only).
        """
        return render_template("dashboard.html")

    # NOTE:
    # Future API endpoints (for AI model, data queries, recognition, etc.)
    # should be added here, for example:
    #
    # @app.route("/api/predict", methods=["OST"])
    # def api_predict():
    #     # TODO: receive parameters & call ML/LLM services
    #     pass
    #
    # Keep this file lightweight for now since the user only needs static pages.


    return app


if __name__ == "__main__":
    # Development entry-point. In production, prefer a WSGI server.
    app = create_app()
    app.run(debug=True)



from flask import Flask, render_template, request

from querysearch import querysearch

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        results = querysearch(query)
        return render_template("results.html", query=query, results=results[:20])
    else:
        return render_template("index.html")

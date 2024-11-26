import flask

app = flask.Flask(__name__)

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    item_name = flask.request.form.get('Item_name')
    return f"Yo this working??? Pulling data for... {item_name}"

if __name__ == "__main__":
    app.run()
'''Imports'''
from flask import Flask, redirect, url_for, render_template, request # Web app framework
from flask_sqlalchemy import SQLAlchemy # Abstracting database interface

import os # File manipulation
import shutil # Deleting directories recursively

import json # Encoding and decoding parameters

from subprocess import Popen, PIPE # Running backend


'''App Initialisation'''
app = Flask(__name__) # Initialising flask app

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///trainer.db" # Setting up database as file on disk
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False # For performance, irrelevant to this project
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disabling caching so model history plot image shows up correctly
db = SQLAlchemy(app) # Initialising database interface


'''Database Definitions'''
# Datasets Table
class Datasets(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    track = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.String(200))
    
    models = db.relationship("Models", backref="datasets", lazy=True) # Bi-directional one-to-many relationship with backref
    
    def __repr__(self):
        return "<Name %r>" % self.id

# Models Table
class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    parameters = db.Column(db.String(500), nullable=False)
    comment = db.Column(db.String(200))
    
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False) # Foreign key
    
    def __repr__(self):
        return "<Name %r>" % self.id


'''Routes'''
# Homepage
@app.route("/")
def home():
    return render_template("home.html")

# Dataset Manager
@app.route("/datasets")
def datasets():
    datasets = Datasets.query.order_by(Datasets.name) # Query all datasets from table
    return render_template("datasets.html", datasets=datasets)

# Add Dataset
@app.route("/add_dataset", methods=["POST", "GET"])
def add_dataset():
    dataset_names = [f.path.split("/")[1] for f in os.scandir("datasets") if f.is_dir()] # Reading all immediate subfolders in the "datasets" directory
    
    if request.method == "POST":
        dataset_name = request.form["name"]
        dataset_track = request.form["track"]
        dataset_comment = request.form["comment"]
        
        new_dataset = Datasets(name=dataset_name, track=dataset_track, comment=dataset_comment) # Creating new record
        
        try:            
            db.session.add(new_dataset)
            db.session.commit()
            
            if dataset_name not in dataset_names:
                os.makedirs("datasets/" + dataset_name) # Create empty directory if it does not exist already
            
            return redirect("/datasets")
        except:
            return "Error adding dataset"
    else:
        return render_template("add_dataset.html", folders=dataset_names)

# Edit Dataset
@app.route("/edit_dataset/<int:id>", methods=["POST", "GET"])
def edit_dataset(id):
    dataset_names = [f.path.split("/")[1] for f in os.scandir("datasets") if f.is_dir()]
    dataset_to_update = Datasets.query.get_or_404(id) # Grabbing record by ID
    old_dataset_name = dataset_to_update.name
    
    if request.method == "POST":
        dataset_to_update.name = request.form["name"]
        dataset_to_update.track = request.form["track"]
        dataset_to_update.comment = request.form["comment"]
        
        try:
            if old_dataset_name in dataset_names:
                os.rename("datasets/" + old_dataset_name, "datasets/" + request.form["name"]) # Renaming directory to follow database
            
            db.session.commit()
            
            return redirect("/datasets")
        except:
            return "Error updating dataset"
    else:
        return render_template("edit_dataset.html", folders=dataset_names, dataset_to_update=dataset_to_update)
        
# Delete Dataset
@app.route("/delete_dataset/<int:id>")
def delete_dataset(id):
    dataset_names = [f.path.split("/")[1] for f in os.scandir("datasets") if f.is_dir()]
    dataset_to_delete = Datasets.query.get_or_404(id)
    
    try:
        if dataset_to_delete.name in dataset_names:
            shutil.rmtree("datasets/" + dataset_to_delete.name) # Recursively deletes folder and all files inside

        Datasets.query.filter_by(id=id).delete()
        db.session.commit()

        return redirect("/datasets")
    except:
        return "Error deleting dataset"

# Training Wizard
@app.route("/train/<int:id>", methods=["POST", "GET"])
def train(id):
    dataset_to_train = Datasets.query.get_or_404(id)
    
    if request.method == "POST":
        config = request.form.to_dict().copy()
        config["data_dir"] = "datasets/" + dataset_to_train.name # Adding key/value pair to dictionary for data directory
        
        # Generating model by name scheme: [origin dataset name]_[incremental number].h5
        free_model_name_found = False
        append_num = 1
        models = os.listdir("models")
        while free_model_name_found == False:
            model_name = dataset_to_train.name + "_" + str(append_num) + ".h5"
            if model_name in models:
                append_num += 1
            else:
                free_model_name_found = True
            config["model_dir"] = "models/" + model_name
        
        # Writing parameters as JSON into text file
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        # Opening process and grabbing output once it is done running, this may take a long time depending on training parameters
        p = Popen(['python3', '-u', 'backend.py'], stdout=PIPE)
        raw_out, _ = p.communicate()
        output = raw_out.decode("utf-8")
        output_lines = output.split("\n")

        new_model = Models(name=model_name, dataset_id=dataset_to_train.id, parameters=json.dumps(config), comment="") # Creating new record
        try:            
            db.session.add(new_model)
            db.session.commit()
            
            return render_template('training_result.html',output=output_lines)
        except:
            return "Error adding model"
    else:
        return render_template("training_setup.html", dataset_to_train=dataset_to_train)

# Model Manager
@app.route("/models")
def models():
    models = db.session.query(Models, Datasets).join(Datasets) # Using Left Join by Models to get the parent dataset via the relationship defined in the database
    return render_template("models.html", models=models)

# Edit Model
@app.route("/edit_model/<int:id>", methods=["POST", "GET"])
def edit_model(id):
    model_names = [f.path.split("/")[1] for f in os.scandir("models")] # Searching all files in the models directory
    model_to_update, parent_dataset = db.session.query(Models, Datasets).filter_by(id=id).join(Datasets).first() # Getting the desired model and it's only parent dataset (since it is a 1-many relationship)
    old_model_name = model_to_update.name
    
    if request.method == "POST":
        model_to_update.name = request.form["name"]
        model_to_update.comment = request.form["comment"]
        
        try:
            if old_model_name in model_names:
                os.rename("models/" + old_model_name, "models/" + request.form["name"]) # Renaming the model file
            
            db.session.commit()
            
            return redirect("/models")
        except:
            return "Error updating model"
    else:
        params = json.loads(model_to_update.parameters) # Converting from JSON into Python dictionary
        return render_template("edit_model.html", files=model_names, model_to_update=model_to_update, parent_dataset=parent_dataset, params=params)

# Delete Model
@app.route("/delete_model/<int:id>")
def delete_model(id):
    model_names = [f.path.split("/")[1] for f in os.scandir("models")]
    model_to_delete = Models.query.get_or_404(id)
    
    try:
        if model_to_delete.name in model_names:
            os.remove("models/" + model_to_delete.name) # Deleting the model file
        
        Models.query.filter_by(id=id).delete() # Removing the record in the database
        db.session.commit()
        
        return redirect("/models")
    except:
        return "Error deleting model"
      

'''Program'''
if __name__ == "__main__":
    app.run(debug=True)
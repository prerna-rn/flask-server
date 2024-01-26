from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import os
import io
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://erpportal:erpcsd2022@cluster0.13toh.mongodb.net/erpportal?retryWrites=true&w=majority"
db = PyMongo(app).db

@app.route('/students/studentMarks')

def perform_clustering_and_export():
    # Load the CSV file containing both training and test data
    file_name = 'student_feedback_data1.csv'
    base_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, file_name)
    data = pd.read_csv(file_path)

    # Preparing the data for clustering
    X = data[['student_feedback', 'subj_teacher_feedback', 'overall_feedback']]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Add a new column to the data frame to store cluster labels
    data['predicted_cluster'] = kmeans.predict(X)

    # Identify beginner, intermediate, and advanced learners based on cluster centroids
    centroids = kmeans.cluster_centers_

    for i, centroid in enumerate(centroids):
        feedback = centroid[2]  # considering overall_feedback from centroid
        if feedback < 4:
            print(f"Cluster {i}: Beginner Learners")
        elif 4 <= feedback < 7:
            print(f"Cluster {i}: Intermediate Learners")
        else:
            print(f"Cluster {i}: Advanced Learners")

    # Export the output clusters (complete row) to a CSV file
    output_file = 'output_clusters.csv'  # Replace with your desired output file path
    data.to_csv(output_file, index=False)

    # Drop unnecessary columns before converting to JSON
    data_to_export = data[['student_id', 'predicted_cluster']]

    # Convert the DataFrame to JSON format
    json_data = data_to_export.to_json(orient='records')

    # Use io.StringIO as a context manager to read the JSON data
    with io.StringIO(json_data) as json_io:
    # Upload JSON data to MongoDB
        db.clusters.insert_many(pd.read_json(json_io, orient='records').to_dict('records'))

    print("Data uploaded to MongoDB successfully.")

# Call the function to perform clustering, export output clusters to CSV, and then upload to MongoDB
perform_clustering_and_export()

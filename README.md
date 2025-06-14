# MACHINE-LEARNING-MODEL-IMPLEMENTATION-1

COMPANY: COOTECH IT SOLUTIONS

NANE: sharan ganesh konakala

INTERN ID: CT06DN445

DONAIN: Python Programming

DURATION: 6 WEEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION

In this task, I successfully implemented a machine learning model for spam message classification using Python and key libraries such as pandas, scikit-learn, joblib, and Streamlit. The primary objective was to build a spam detection system capable of identifying whether a given SMS message is spam or not. The first step in the process involved acquiring and loading the dataset, which was a CSV file containing labeled SMS messages, downloaded from Kaggle. Using pandas, I loaded the dataset and performed preprocessing by dropping unnecessary columns, renaming them for clarity, and converting categorical labels like "ham" and "spam" into numerical form (0 for ham and 1for spam), making them suitable for machine learning algorithms.

After preprocessing, I used scikit-learn’s TfidfVectorizer to convert the text messages into numerical feature vectors based on Term Frequency–Inverse Document Frequency, which helps in understanding the importance of words in messages relative to the corpus. These vectors were then split into training and testing sets using the train_test_split function to ensure the model was evaluated fairly. For classification, I used a Logistic Regression model from scikit-learn, which is a well-suited algorithm for binary classification problems like spam detection. After training the model, I evaluatedits performance using accuracy and a classification report that included precision, recall, and F1-score. The model achieved a high accuracy of approximately 97.8%, demonstrating its effectiveness in correctly classifying spam and non-spam messages.

Once the model was trained and evaluated, I serialized the model and vectorizer using joblib, saving them as .pkl files (spam_classifier_model.pkl and tfidf_vectorizer.pkl). This step ensured that the model could be reused without retraining, enabling efficient deployment. For the user interface, I developed a simple yet interactive web application using Streamlit, a Python library designed for creating web appsfor machine learning and data science projects. The app included a title, a text area for users to input messages, and a button to trigger the prediction. Upon clicking the button, the input message is transformed using the saved TF-IDF vectorizer, and the loaded model predicts whether the message is spam or not. The result is then displayed to the user with a clear label—either "🔴 Spam" or "🟢 Ham".

To run the app, I used the command streamlit run spam_app.py in the terminal. The app opened in a browser and successfully allowed real-time testing of the spam classifier. Throughout the project, I also handled potential issues like missing data,warnings related to metrics in scikit-learn, and ensured a clean separation between model training and deployment. This task showcased a full machine learning workflow—from data preprocessing and model training to model evaluation, serialization, and front-end deployment using Streamlit. It provided practical experience in building a functional ML pipeline, aligning with real-world applications where model performance and usability are both crucial. This project is a significant step in my journey toward mastering applied machine learning and deploying intelligent applications.

OUTPUT

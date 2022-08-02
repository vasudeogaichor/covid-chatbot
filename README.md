# COVID CHATBOT
This project uses the neural networks and transformer architecture in order to classify user input and give reply accordingly from datasets.

## HOW TO RUN THE PROJECT
### 1. Create a python virtual environment and activate it
> python -m venv venv
> source ~/venv/bin/activate

### 2. Install the required libraries using the requirements.txt file
> pip install -r requirements.txt

### 3. Create a basic chatbot model with neural networks and bag of words
> python create_chatbot_model.py

### 4. Clean the data before encoding it
> python create_encoder_data.py

### 5. Create embeddings for coronavirus related questions using sentence transformer model
> python question_embeddings.py

### 6. Run the flask server and GUI for chatbot simultaneously
> python flask_app.py

> python chatbot_gui.py
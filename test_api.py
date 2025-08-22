import requests
import json
import logging

# Configure basic logging to see the output
logging.basicConfig(level=logging.INFO)

# Define the API endpoint URL
API_URL = "http://127.0.0.1:5000/predict"

def test_api():
    """
    Sends a sample JSON payload to the deployed API endpoint and prints the response.
    """
    # Sample data for a passenger from the test set
    # This example mimics a passenger who might have a high chance of survival
    sample_data = {
        "PassengerId": 900,
        "Pclass": 1,
        "Name": "Miss. Elizabeth Gladys 'Bess' Dean",
        "Sex": "female",
        "Age": 33.0,
        "SibSp": 1,
        "Parch": 1,
        "Fare": 90.0,
        "Embarked": "S"
    }
    
    # Another sample, for a male passenger with lower class and fare
    sample_data_2 = {
        "PassengerId": 901,
        "Pclass": 3,
        "Name": "Mr. John Smith",
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.0,
        "Embarked": "Q"
    }

    try:
        logging.info("Sending prediction request for Sample 1...")
        response_1 = requests.post(API_URL, json=sample_data)
        response_1.raise_for_status() # Raise an exception for bad status codes
        
        logging.info("Sample 1 Response Status: %s", response_1.status_code)
        logging.info("Sample 1 Response Data: %s", json.dumps(response_1.json(), indent=2))
        
        logging.info("\nSending prediction request for Sample 2...")
        response_2 = requests.post(API_URL, json=sample_data_2)
        response_2.raise_for_status()
        
        logging.info("Sample 2 Response Status: %s", response_2.status_code)
        logging.info("Sample 2 Response Data: %s", json.dumps(response_2.json(), indent=2))

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while connecting to the API: {e}")

if __name__ == "__main__":
    test_api()

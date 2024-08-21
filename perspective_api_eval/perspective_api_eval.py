import requests
import concurrent.futures
import time
import os

PERSPECTIVE_API_KEY = os.environ['PERSPECTIVE_API_KEY']
PERSPECTIVE_API_URL = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'
MAX_BATCH_SIZE = 100  # Example batch size, adjust based on limits
MAX_CONCURRENT_REQUESTS = 10  # Number of parallel requests


def evaluate_toxicity(batch):
    request_body = {
        'comment': {'text': batch},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}},
    }
    params = {'key': PERSPECTIVE_API_KEY}
    response = requests.post(PERSPECTIVE_API_URL, json=request_body, params=params)
    return response.json()


def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def main():
    data = load_data()  # Load your list of model generations here

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = []
        for batch in batch_generator(data, MAX_BATCH_SIZE):
            futures.append(executor.submit(evaluate_toxicity, batch))
            time.sleep(0.1)  # Sleep to manage API rate limits

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            process_result(result)  # Save or process the API response


def load_data():
    # Replace this with your actual data loading logic
    return 'test hello!'
    #return ['Generation 1', 'Generation 2', 'test fuck! generation', 'Generation 800000']


def process_result(result):
    # Implement this to store or process the API results
    print("Result: ", result)



if __name__ == '__main__':
    main()

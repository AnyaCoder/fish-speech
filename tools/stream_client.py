import requests

def fetch_stream():
    response = requests.post("http://127.0.0.1:8000/stream/default", stream=True)

    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None):
            print(chunk.decode())

if __name__ == "__main__":
    fetch_stream()
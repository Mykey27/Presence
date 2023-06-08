import requests

def download_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filename, "wb") as file:
            file.write(response.content)
            
        print("Image downloaded successfully.")
        
    except requests.exceptions.RequestException as e:
        print("Error:", e)

# Example usage

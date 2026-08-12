import requests
import os

# 1. إنشاء صورة وهمية
dummy_image = "test_dummy.jpg"
with open(dummy_image, "wb") as f:
    f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')

url = "http://127.0.0.1:5000"

print("="*40)
print("Test 1: POST /analyze/image (Upload File)")
print("="*40)
with open(dummy_image, "rb") as img:
    files = {"image": ("test_image.jpg", img, "image/jpeg")}
    response = requests.post(f"{url}/analyze/image", files=files)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}\n")

print("="*40)
print("Test 2: GET /results")
print("="*40)
response2 = requests.get(f"{url}/results")
print(f"Status Code: {response2.status_code}")
print(f"Response: {response2.json()}\n")

if os.path.exists(dummy_image):
    os.remove(dummy_image)
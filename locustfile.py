from locust import HttpUser, task, between
import random
import os
import config

class TelegramBotUser(HttpUser):
    host = "https://api.telegram.org"
    wait_time = between(3, 7)  # Увеличенный интервал
    
    @task
    def send_photo(self):
        test_images = os.listdir("test_images")
        if not test_images:
            print("No test images found!")
            return
            
        img_path = random.choice(test_images)
        
        try:
            with open(f"test_images/{img_path}", "rb") as f:
                response = self.client.post(
                    f"/bot{config.TOKEN}/sendPhoto",
                    files={"photo": (img_path, f, "image/jpeg")},
                    data={"chat_id": 810874650},
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"Failed to send {img_path}: {str(e)}")
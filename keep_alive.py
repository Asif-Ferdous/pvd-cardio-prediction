import requests
import time
import schedule
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("keep_alive.log"),
        logging.StreamHandler()
    ]
)

# Get app URL from environment or use default
APP_URL = os.environ.get('APP_URL', 'https://your-render-app-url.onrender.com')
PING_INTERVAL = int(os.environ.get('PING_INTERVAL_MINUTES', 10))

def ping_app():
    """Send a request to the app's test endpoint to keep it alive"""
    try:
        start_time = time.time()
        response = requests.get(f"{APP_URL}/test")
        end_time = time.time()
        
        if response.status_code == 200:
            logging.info(f"Ping successful! Response time: {end_time - start_time:.2f}s")
        else:
            logging.warning(f"Ping failed with status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error pinging application: {str(e)}")

def main():
    """Main function to schedule and run pings"""
    logging.info(f"Keep-alive service started. Will ping {APP_URL} every {PING_INTERVAL} minutes")
    
    # Schedule the ping job
    schedule.every(PING_INTERVAL).minutes.do(ping_app)
    
    # Do an initial ping
    ping_app()
    
    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
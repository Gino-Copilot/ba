import time
import random
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager


def random_browsing():
    """
    Opens random websites from a list and navigates randomly from one to the next.
    """
    # List of top 20 banned websites (example sites - can be adjusted)
    urls = [
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.voanews.com",
        "https://www.rferl.org",
        "https://www.dw.com",
        "https://www.reddit.com",
        "https://www.facebook.com",
        "https://www.instagram.com",
        "https://www.twitter.com",
        "https://www.tiktok.com",
        "https://www.amnesty.org",
        "https://www.hrw.org",
        "https://www.nobelprize.org",
        "https://www.cfr.org",
        "https://www.freetibet.org",
        "https://www.savetibet.org",
        "https://www.hrw.org",
        "https://www.un.org",
        "https://www.greenpeace.org",
        "https://www.icrc.org"
    ]

    # Configuration for Firefox WebDriver
    options = Options()
    # Uncomment the following line for headless mode
    # options.headless = True

    # Initialize WebDriver with GeckoDriverManager
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    try:
        print("Starting random browsing...")
        for _ in range(20):  # Visits 20 pages (can be adjusted)
            url = random.choice(urls)
            print(f"Visiting: {url}")
            driver.get(url)

            # Random dwell time between 2 and 10 seconds
            wait_time = random.randint(2, 10)
            print(f"Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
    finally:
        driver.quit()
        print("Traffic generation completed.")


if __name__ == "__main__":
    random_browsing()

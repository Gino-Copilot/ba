import os
import subprocess
import time
import random
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from configure_proxy import configure_proxy  # Importiere die Proxy-Konfiguration


def ensure_non_root():
    """
    Ensures that the script is not running as root.
    """
    if os.geteuid() == 0:
        # Relaunch script as 'gino'
        user = "gino"
        python_executable = os.environ.get("VIRTUAL_ENV") + "/bin/python"
        script_path = os.path.abspath(__file__)
        print(f"Relaunching script as user: {user}")
        subprocess.run(["sudo", "-u", user, python_executable, script_path])
        exit()


def visit_my_ip(driver):
    """
    Opens the 'Meine IP' page to check the current IP address.
    """
    print("Visiting 'Meine IP' page...")
    url = "https://whatismyipaddress.com/de/meine-ip#google_vignette"
    try:
        driver.get(url)
        time.sleep(5)  # Wait for 5 seconds to ensure the page fully loads
        print("Stayed on 'Meine IP' page for 5 seconds.")
    except Exception as e:
        print(f"Error visiting 'Meine IP' page: {e}")


def random_browsing(driver):
    """
    Opens random websites from a list and navigates randomly from one to the next.
    """
    # List of websites to visit
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
        "https://www.un.org",
        "https://www.greenpeace.org",
        "https://www.icrc.org"
    ]

    while True:  # Infinite loop
        url = random.choice(urls)
        print(f"Visiting: {url}")
        driver.get(url)

        # Random dwell time between 2 and 10 seconds
        wait_time = random.uniform(2, 10)
        print(f"Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time)



def main():
    # Ensure script is not running as root
    ensure_non_root()

    # Configure Proxy
    proxy = configure_proxy()

    # Configuration for Firefox WebDriver
    options = Options()
    options.headless = False  # Make browser visible
    options.binary_location = "/usr/bin/firefox"  # Adjust binary location if needed
    options.proxy = proxy  # Apply proxy configuration

    # Initialize WebDriver with GeckoDriverManager
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    try:
        # Step 1: Visit 'Meine IP' page
        visit_my_ip(driver)

        # Step 2: Start random browsing
        random_browsing(driver)
    finally:
        driver.quit()
        print("Traffic generation completed.")


if __name__ == "__main__":
    main()

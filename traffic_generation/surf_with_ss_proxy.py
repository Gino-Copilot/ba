import os
import subprocess
import time
import random
import traceback

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

from configure_proxy import configure_proxy  # proxy module


def ensure_non_root():
    """
    Ensures that the script is not running as root.
    If root, re-launch as a non-root user (here 'gino') and exit.
    """
    if os.geteuid() == 0:
        user = "gino"  # Please make sure this user actually exists on your system
        python_executable = os.environ.get("VIRTUAL_ENV") + "/bin/python"
        script_path = os.path.abspath(__file__)
        print(f"Relaunching script as user: {user}")
        subprocess.run(["sudo", "-u", user, python_executable, script_path])
        exit()


def visit_my_ip(driver):
    """
    Opens the IP check page for 5 seconds.
    """
    print("Visiting IP check page...")
    url = "https://whatismyipaddress.com/de/meine-ip#google_vignette"
    try:
        driver.get(url)
        time.sleep(5)
        print("Stayed on IP check page for 5 seconds.")
    except Exception as e:
        print(f"Error visiting IP check page: {e}")


def random_browsing(driver):
    """
    Randomly visits a list of URLs and waits for a random time in between.
    This loop is intended to run forever.
    """
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

    while True:
        try:
            url = random.choice(urls)
            print(f"Visiting: {url}")
            driver.get(url)

            wait_time = random.uniform(2, 10)
            print(f"Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Error during random browsing on {url}: {e}")
            # Print full traceback for debugging
            traceback.print_exc()
            # You can either break or continue here:
            # 'break' -> leaves the while True in random_browsing
            # 'continue' -> retries the loop
            # For a complete restart from run_forever(), use break:
            break


def main():
    """
    Main function that sets up the browser, checks IP, and starts random browsing.
    """
    ensure_non_root()

    # Get proxy object
    proxy = configure_proxy()

    # Firefox options
    options = Options()
    options.headless = False

    # Bind proxy to Selenium-Firefox-driver
    options.proxy = proxy

    # Basic proxy settings
    options.set_preference("network.proxy.type", 1)
    options.set_preference("network.proxy.socks", "127.0.0.1")
    options.set_preference("network.proxy.socks_port", 1080)

    # If you want DNS queries to go through SOCKS5 (encrypted), set this to True
    # Otherwise, DNS is done locally
    options.set_preference("network.proxy.socks_remote_dns", False)

    # Initialize WebDriver
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    try:
        # Check IP
        visit_my_ip(driver)

        # Random browsing (loop runs indefinitely)
        random_browsing(driver)

    finally:
        # Quit the driver if we exit random_browsing or if an exception happens
        driver.quit()
        print("Traffic generation completed (driver closed).")


def run_forever():
    """
    Calls main() in an infinite loop and restarts if an error occurs or if main() finishes unexpectedly.
    """
    while True:
        try:
            main()
        except Exception as e:
            print("********** AN ERROR OCCURRED **********")
            print(f"Error message: {e}")
            print("Stacktrace:")
            traceback.print_exc()
            print("***************************************")
            time.sleep(5)  # wait a bit before restarting
        else:
            # If main() returned normally, that means random_browsing ended somehow
            print("No exception caught, but main() ended unexpectedly. Restarting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    run_forever()

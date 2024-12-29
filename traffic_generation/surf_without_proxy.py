import os
import subprocess
import time
import random
import traceback

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from webdriver_manager.firefox import GeckoDriverManager

def ensure_non_root():
    """
    Ensures the script is not running as root.
    If it is, relaunch as a non-root user (here 'gino') and exit.
    """
    if os.geteuid() == 0:
        user = "gino"
        python_executable = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python")
        script_path = os.path.abspath(__file__)
        print(f"Relaunching script as user: {user}")
        subprocess.run(["sudo", "-u", user, python_executable, script_path])
        exit()

def visit_my_ip(driver):
    """
    Opens an IP check page for 5 seconds to verify the current IP address.
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
    Randomly visits a list of URLs in an infinite loop.
    Waits a random amount of time (2-10 seconds) between visits.
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
            traceback.print_exc()
            # Break out so that run_forever can restart main().
            break

def main():
    """
    Main function:
    1. Ensures we are not root.
    2. Launches Firefox with a clean profile (no proxy).
    3. Visits an IP check page, then performs random browsing in an infinite loop.
    """
    ensure_non_root()

    # Create a fresh Firefox profile to avoid any leftover settings.
    fresh_profile = FirefoxProfile()
    # Ensure default proxy settings: 0 = direct connection (no proxy).
    fresh_profile.set_preference("network.proxy.type", 0)

    options = Options()
    options.headless = False
    # If you need a specific Firefox binary location, uncomment and adjust:
    # options.binary_location = "/usr/bin/firefox"

    # Attach the fresh profile to the driver options
    options.profile = fresh_profile

    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    try:
        visit_my_ip(driver)
        random_browsing(driver)
    finally:
        driver.quit()
        print("Traffic generation completed (driver closed).")

def run_forever():
    """
    Calls main() in an infinite loop.
    If an error or a break occurs, wait 5 seconds and restart.
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
            time.sleep(5)
        else:
            print("No exception caught, but main() ended. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_forever()

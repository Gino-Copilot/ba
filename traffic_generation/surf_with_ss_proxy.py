import os
import subprocess
import time
import random

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

from configure_proxy import configure_proxy  # proxy module


def ensure_non_root():
    """
    Ensures that the script is not running as root.
    """
    if os.geteuid() == 0:
        user = "gino"
        python_executable = os.environ.get("VIRTUAL_ENV") + "/bin/python"
        script_path = os.path.abspath(__file__)
        print(f"Relaunching script as user: {user}")
        subprocess.run(["sudo", "-u", user, python_executable, script_path])
        exit()


def visit_my_ip(driver):
    """
    Opens the IP check page.
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
    Opens random websites from the list and navigates.
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
        url = random.choice(urls)
        print(f"Visiting: {url}")
        driver.get(url)

        wait_time = random.uniform(2, 10)
        print(f"Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time)


def main():
    ensure_non_root()

    # Get proxy object
    proxy = configure_proxy()

    # Firefox options
    options = Options()
    options.headless = False

    # Bind proxy to selenium-firefox-driver
    options.proxy = proxy

    options.set_preference("network.proxy.type", 1)

    # Manual SOCKS configuration
    options.set_preference("network.proxy.socks", "127.0.0.1")
    options.set_preference("network.proxy.socks_port", 1080)

    # Remote DNS configuration
    # True: DNS queries through SOCKS5 proxy (encrypted via Shadowsocks)
    # False: DNS queries locally
    options.set_preference("network.proxy.socks_remote_dns", False)

    # HTTP/HTTPS proxy settings (optional for SOCKS5):
    # options.set_preference("network.proxy.http", "127.0.0.1")
    # options.set_preference("network.proxy.http_port", 1080)

    # Initialize WebDriver
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    try:
        # Check IP
        visit_my_ip(driver)

        # Random browsing
        random_browsing(driver)
    finally:
        driver.quit()
        print("Traffic generation completed.")


if __name__ == "__main__":
    main()
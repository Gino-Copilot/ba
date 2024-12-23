import os
import subprocess
import time
import random

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

from configure_proxy import configure_proxy  # Dein Proxy-Modul


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
    Opens the 'Meine IP' page to check the current IP address.
    """
    print("Visiting 'Meine IP' page...")
    url = "https://whatismyipaddress.com/de/meine-ip#google_vignette"
    try:
        driver.get(url)
        time.sleep(5)
        print("Stayed on 'Meine IP' page for 5 seconds.")
    except Exception as e:
        print(f"Error visiting 'Meine IP' page: {e}")


def random_browsing(driver):
    """
    Opens random websites from a list and navigates randomly.
    """
    urls = [
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.reddit.com",
        "https://www.facebook.com",
        "https://www.instagram.com",
        "https://www.twitter.com",
        # usw.
    ]

    while True:  # Endlosschleife
        url = random.choice(urls)
        print(f"Visiting: {url}")
        driver.get(url)

        # Warte zwischen 2 und 10 Sekunden
        wait_time = random.uniform(2, 10)
        print(f"Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time)


def main():

    ensure_non_root()

    # 2 get the proxy Object
    proxy = configure_proxy()

    # 3) Ffirefox-options
    options = Options()
    options.headless = False

    # bind proxy to selenium-chrome-driver
    options.proxy = proxy


    options.set_preference("network.proxy.type", 1)

    # b) Manuell für SOCKS definieren (redundant, weil wir Selenium Proxy setzen)
    options.set_preference("network.proxy.socks", "127.0.0.1")
    options.set_preference("network.proxy.socks_port", 1080)

    # c) Remote DNS => True oder False
    # Wenn du "True" einstellst, gehen DNS-Anfragen ebenfalls durch den SOCKS5-Proxy
    # (sprich: verschlüsselt über Shadowsocks).
    # Setz "False", wenn du DNS-Anfragen lokal sehen willst.
    options.set_preference("network.proxy.socks_remote_dns", False)

    # Falls du HTTP/HTTPS-Proxy-Settings brauchst (normalerweise nicht für SOCKS5):
    # options.set_preference("network.proxy.http", "127.0.0.1")
    # options.set_preference("network.proxy.http_port", 1080)
    # ...

    # 4) Starte WebDriver
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    try:
        # Schritt 1: Meine IP anzeigen
        visit_my_ip(driver)

        # Schritt 2: Random Browsing
        random_browsing(driver)
    finally:
        driver.quit()
        print("Traffic generation completed.")


if __name__ == "__main__":
    main()

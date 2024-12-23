from selenium.webdriver.common.proxy import Proxy, ProxyType

def configure_proxy():
    """
    Configures the Firefox WebDriver to use the Shadowsocks SOCKS5 proxy.
    Returns a configured Proxy object.
    """
    # 1) Erstelle ein Proxy-Objekt
    proxy = Proxy()
    proxy.proxy_type = ProxyType.MANUAL

    # 2) SOCKS5-Einstellungen
    proxy.socks_proxy = "127.0.0.1:1080"  # Shadowsocks-Client läuft lokal
    proxy.socks_version = 5              # SOCKS5

    # 3) Keine Domains bypassen
    proxy.no_proxy = ""

    return proxy

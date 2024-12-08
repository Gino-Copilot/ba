from selenium.webdriver.common.proxy import Proxy, ProxyType

def configure_proxy():
    """
    Configures the Firefox WebDriver to use the Shadowsocks proxy.
    Returns a configured Proxy object.
    """
    proxy = Proxy()
    proxy.proxy_type = ProxyType.MANUAL
    proxy.socks_proxy = "127.0.0.1:1080"  # Proxy address and port
    proxy.socks_version = 5  # SOCKS5
    proxy.no_proxy = ""  # Do not bypass proxy for any domain
    return proxy

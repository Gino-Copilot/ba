import subprocess
import time
from datetime import datetime
import os
import signal
import psutil

class TrafficCapture:
    def __init__(self, traffic_type="default_traffic", interval=10, file_count=200, interface="wlp0s20f3"):
        # Spezifischer Unterordner für die Art des Traffics im traffic_data-Verzeichnis
        base_output_dir = os.path.join(os.path.dirname(__file__), '..', 'traffic_data')
        self.output_dir = os.path.join(base_output_dir, traffic_type)
        self.interval = interval
        self.file_count = file_count
        self.interface = interface
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Stellt sicher, dass das Ausgabe-Verzeichnis existiert."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """Überprüft den verfügbaren Speicherplatz im Ausgabe-Verzeichnis."""
        disk = psutil.disk_usage(self.output_dir)
        if disk.percent > 80:
            print(f"Warnung: Nur noch {disk.free / (1024 * 1024 * 1024):.2f} GB freier Speicherplatz")

    def capture_traffic(self):
        """Startet die Traffic-Erfassung."""
        self._check_disk_space()

        for i in range(self.file_count):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(self.output_dir, f"traffic_capture_{timestamp}.pcap")

            command = [
                "sudo", "tcpdump",
                "-i", self.interface,
                "-w", output_file,
                "-G", str(self.interval),
                "-W", str(self.file_count),
                "-s", "0",
                "-B", "2048",
                "-n",
                "-K",
                "-q",
                "-Z", "root",
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                time.sleep(self.interval)

                if process.poll() is not None:
                    _, stderr = process.communicate()
                    print(f"tcpdump beendet mit Fehler: {stderr.decode()}")
                    continue

                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)

                file_size = os.path.getsize(output_file) / (1024 * 1024)
                print(f"Saved traffic to {output_file} ({file_size:.2f} MB)")

            except subprocess.TimeoutExpired:
                print("Timeout beim Beenden von tcpdump")
                process.kill()
            except Exception as e:
                print(f"Error during capturing: {e}")
                break

    def cleanup(self):
        """Beendet alle tcpdump-Prozesse, um das System sauber zu halten."""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'tcpdump':
                proc.kill()

def main():
    # Beispiel: ProtonVPN-Traffic speichern
    traffic_capture = TrafficCapture(traffic_type="protonvpn_traffic", interval=10, file_count=200, interface="wlp0s20f3")

    try:
        traffic_capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
    finally:
        traffic_capture.cleanup()

if __name__ == "__main__":
    main()

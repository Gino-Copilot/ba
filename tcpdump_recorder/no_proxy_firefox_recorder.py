import subprocess
import time
from datetime import datetime
import os
import signal
import psutil


class TrafficCapture:
    def __init__(self, traffic_type="default_traffic", interval=10, file_count=200, interface="wlp0s20f3"):
        # Add date to the traffic type for folder naming
        current_date = datetime.now().strftime("%Y-%m-%d")
        folder_name = f"{traffic_type}_{current_date}"

        # Specific subfolder for traffic type in traffic_data directory
        base_output_dir = os.path.join(os.path.dirname(__file__), '..', 'traffic_data')
        self.output_dir = os.path.join(base_output_dir, folder_name)

        self.interval = interval
        self.file_count = file_count
        self.interface = interface
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Make sure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """Check available disk space in output directory."""
        disk = psutil.disk_usage(self.output_dir)
        if disk.percent > 80:
            print(f"Warning: Only {disk.free / (1024 * 1024 * 1024):.2f} GB free disk space left")

    def capture_traffic(self):
        """Start traffic capture."""
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
                    print(f"tcpdump terminated with error: {stderr.decode()}")
                    continue

                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)

                file_size = os.path.getsize(output_file)  # File size in bytes
                print(f"Saved traffic to {output_file} ({file_size} Bytes)")

            except subprocess.TimeoutExpired:
                print("Timeout while stopping tcpdump")
                process.kill()
            except Exception as e:
                print(f"Error during capturing: {e}")
                break

    def cleanup(self):
        """Stop all tcpdump processes to keep the system clean."""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'tcpdump':
                proc.kill()


def main():
    # Example: Save Firefox browser traffic without proxy
    traffic_capture = TrafficCapture(traffic_type="firefox_without_proxy", interval=3, file_count=200,
                                     interface="wlp0s20f3")

    try:
        traffic_capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProgram is shutting down...")
    finally:
        traffic_capture.cleanup()


if __name__ == "__main__":
    main()

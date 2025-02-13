import subprocess
import os
import time
from datetime import datetime
import psutil
import signal

class TrafficCapture:
    def __init__(self, traffic_type="shadowsocks_traffic", interval=10, file_count=100, interface="wlp0s20f3", port=8388):
        # Include date in folder name
        current_date = datetime.now().strftime("%m-%d")
        folder_name = f"{traffic_type}_{current_date}"
        base_output_dir = os.path.join(os.path.dirname(__file__), '..', 'traffic_data')
        self.output_dir = os.path.join(base_output_dir, folder_name)
        self.interval = interval
        self.file_count = file_count
        self.interface = interface
        self.port = port
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Make sure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """Check available disk space in output directory."""
        disk = psutil.disk_usage(self.output_dir)
        free_space_gb = disk.free / (1024 * 1024 * 1024)  # Convert bytes to GB
        if free_space_gb < 1:  # Warn if less than 1GB is available
            print(f"Warning: Only {free_space_gb:.2f} GB free disk space left")
        return free_space_gb

    def capture_traffic(self):
        """Start traffic capture."""
        free_space_gb = self._check_disk_space()
        if free_space_gb < 1:
            print("Insufficient disk space. Aborting traffic capture.")
            return

        for i in range(1, self.file_count + 1):  # Exact number of files
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")  # Format: MM-DD_HH-MM-SS
            output_file = os.path.join(self.output_dir, f"capture_{i:03d}_{timestamp}.pcap")

            # Tcpdump command for a single file
            command = [
                "sudo", "tcpdump",
                "-i", self.interface,
                "port", str(self.port),
                "-w", output_file,
                "-G", str(self.interval),
                "-s", "0",
                "-B", "8192",  # Increased buffer size
                "-n",
                "-U",          # Immediate write to file
                "-K",          # Disable packet checksum verification
                "-q",
                "-Z", "root",
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                time.sleep(self.interval)  # Wait for the duration of one file

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
    # Save Shadowsocks traffic with updated folder and file names
    traffic_capture = TrafficCapture(
        traffic_type="ss_youtube_selenium_aes-256-gcm",
        interval=5,
        file_count=100,
        interface="wlp0s20f3",
        port=8388
    )

    try:
        traffic_capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProgram is shutting down...")
    finally:
        traffic_capture.cleanup()


if __name__ == "__main__":
    main()

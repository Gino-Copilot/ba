import subprocess
from datetime import datetime
import os
import psutil
import time
import signal

class TrafficCapture:
    def __init__(self, traffic_type="default", interval=3, file_count=5, interface="wlp0s20f3"):
        # Create a unique folder for storing the capture files
        folder_name = f"{traffic_type}_{datetime.now().strftime('%m-%d_%H-%M')}"
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', 'traffic_data', folder_name)
        self.interval = interval
        self.file_count = file_count
        self.interface = interface

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """Checks available disk space and warns if it is low."""
        disk = psutil.disk_usage(self.output_dir)
        if disk.percent > 90:
            print("Warning: Disk space is low!")

    def _verify_file_duration(self, filename):
        """Verifies if the pcap file has been successfully created."""
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"File {filename} created with size: {file_size} bytes")
            if file_size == 0:
                print("Warning: File is empty, tcpdump might not have started correctly.")
        else:
            print(f"Error: File {filename} was not created.")

    def capture_traffic(self):
        """Captures network traffic and writes it to separate pcap files."""
        self._check_disk_space()
        print(f"Starting traffic capture in: {self.output_dir}")

        for i in range(1, self.file_count + 1):
            filename = os.path.join(self.output_dir, f"capture_{i}.pcap")
            print(f"Capturing file {i}/{self.file_count}: {filename}")

            command = [
                "sudo", "tcpdump",
                "-i", self.interface,
                "-w", filename,
                "-s", "0",        # Capture the entire packets
                "-B", "16384",    # Increased buffer size for better performance
                "-n",             # Do not resolve hostnames
                "-q",             # Minimal output
                "--immediate-mode",  # Process packets immediately
                "-Z", "root"      # Run as root
            ]

            try:
                # Start tcpdump process
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                start_time = time.perf_counter()

                # Precisely control the capture duration
                while time.perf_counter() - start_time < self.interval:
                    time.sleep(0.01)  # Fine-grained sleep for accuracy

                # Send SIGINT to ensure clean termination
                process.send_signal(signal.SIGINT)
                process.wait(timeout=2)  # Ensure process termination

            except subprocess.TimeoutExpired:
                print(f"Warning: tcpdump process did not terminate on time. Killing process.")
                process.kill()  # Forcefully terminate process
            except Exception as e:
                print(f"Error during capture: {e}")
                process.kill()
            finally:
                actual_time = time.perf_counter() - start_time
                print(f"File {i} captured in {actual_time:.3f} seconds.")
                self._verify_file_duration(filename)
                self._check_disk_space()

        print("\nTraffic capture completed successfully.")

    def cleanup(self):
        """Terminates any lingering tcpdump processes."""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'tcpdump':
                proc.kill()
        print("Cleaned up all tcpdump processes.")

def main():
    # Example: Capture 5 files, each 3 seconds long
    capture = TrafficCapture(traffic_type="all_traffic", interval=3, file_count=5)
    try:
        capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        capture.cleanup()

if __name__ == "__main__":
    main()

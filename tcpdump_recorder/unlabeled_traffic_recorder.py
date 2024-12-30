# file: unlabeled_traffic_recorder.py

import subprocess
import os
import time
from datetime import datetime
import psutil
import signal

class UnlabeledTrafficCapture:
    def __init__(
        self,
        interval=5,
        file_count=5,
        interface="eth0"
    ):
        """
        This script captures all network traffic on the specified interface (no port/IP filter).
        Each capture runs for 'interval' seconds and is saved to a separate .pcap file.
        A total of 'file_count' files will be created.

        Args:
            interval (int): Number of seconds to capture for each .pcap file.
            file_count (int): Number of .pcap files to create.
            interface (str): The network interface to capture from (e.g., 'eth0' or 'wlan0').
        """

        # Create a base output directory under traffic_data/unlabeled
        base_output_dir = os.path.join(
            os.path.dirname(__file__),
            '..',              # up one level
            'traffic_data',    # main traffic_data folder
            'unlabeled'        # put everything under "unlabeled"
        )

        # Generate a unique subfolder name (e.g. "20231030_153045")
        time_subfolder = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Each script run will thus create a new subdirectory for PCAPs
        self.output_dir = os.path.join(base_output_dir, time_subfolder)
        self.interval = interval
        self.file_count = file_count
        self.interface = interface

        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Make sure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """Check available disk space in output directory."""
        disk = psutil.disk_usage(self.output_dir)
        free_space_gb = disk.free / (1024 * 1024 * 1024)  # Convert bytes to GB
        if free_space_gb < 1:
            print(f"Warning: Only {free_space_gb:.2f} GB free disk space left.")
        return free_space_gb

    def capture_traffic(self):
        """
        Start capturing traffic on the specified interface for 'interval' seconds per file,
        creating 'file_count' total PCAP files with no port or IP filtering.
        """
        free_space_gb = self._check_disk_space()
        if free_space_gb < 1:
            print("Insufficient disk space. Aborting traffic capture.")
            return

        for i in range(1, self.file_count + 1):
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")  # e.g. 10-30_15-05-30
            output_file = os.path.join(self.output_dir, f"capture_{i:03d}_{timestamp}.pcap")

            # Tcpdump command to capture all traffic on the interface
            command = [
                "sudo", "tcpdump",
                "-i", self.interface,
                "-w", output_file,
                "-G", str(self.interval),
                "-s", "0",
                "-B", "8192",
                "-n",
                "-U",
                "-K",
                "-q",
                "-Z", "root"
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Capture traffic for 'interval' seconds
                time.sleep(self.interval)

                # If tcpdump exited prematurely, print error
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    print(f"tcpdump terminated with error: {stderr.decode().strip()}")
                    continue

                # Stop tcpdump cleanly via SIGINT
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)

                file_size = os.path.getsize(output_file)
                print(f"Saved traffic to {output_file} ({file_size} Bytes)")

            except subprocess.TimeoutExpired:
                print("Timeout while stopping tcpdump.")
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
    """
    Example usage:
    Captures 5 files, each 10 seconds long, capturing all traffic on 'eth0'.
    """
    capture = UnlabeledTrafficCapture(
        interval=10,
        file_count=5,
        interface="wlp0s20f3"
    )

    try:
        capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProgram is shutting down (KeyboardInterrupt).")
    finally:
        capture.cleanup()

if __name__ == "__main__":
    main()

import subprocess
import os
import time
from datetime import datetime
import psutil
import signal


class TrafficCapture:
    def __init__(self,
                 traffic_type="normal_traffic",
                 interval=10,
                 file_count=5,
                 interface="wlp0s20f3"):
        """
        Initializes the TrafficCapture object for normal (unencrypted) traffic.

        :param traffic_type: A label to include in the output folder name.
        :param interval: How long (in seconds) each tcpdump capture file should run.
        :param file_count: How many capture files to generate in total.
        :param interface: The network interface to capture from (e.g., 'eth0', 'wlan0', etc.).
        """
        current_date = datetime.now().strftime("%m-%d")
        folder_name = f"{traffic_type}_{current_date}"
        base_output_dir = os.path.join(os.path.dirname(__file__), "..", "traffic_data")
        self.output_dir = os.path.join(base_output_dir, folder_name)
        self.interval = interval
        self.file_count = file_count
        self.interface = interface

        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """
        Ensures the output directory exists.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """
        Checks the available disk space in the output directory and warns if less than 1 GB is free.
        """
        disk = psutil.disk_usage(self.output_dir)
        free_space_gb = disk.free / (1024 * 1024 * 1024)  # bytes to GB
        if free_space_gb < 1:
            print(f"Warning: Only {free_space_gb:.2f} GB free disk space left")
        return free_space_gb

    def capture_traffic(self):
        """
        Starts interval-based traffic capture on the specified interface.
        No port filter is applied, so all traffic on this interface is recorded.
        """
        free_space_gb = self._check_disk_space()
        if free_space_gb < 1:
            print("Insufficient disk space. Aborting traffic capture.")
            return

        for i in range(1, self.file_count + 1):
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            output_file = os.path.join(
                self.output_dir,
                f"capture_{i:03d}_{timestamp}.pcap"
            )

            # The tcpdump command below captures everything (-G splits by interval).
            # No "port" filter is used.
            command = [
                "sudo", "tcpdump",
                "-i", self.interface,
                "-w", output_file,
                "-G", str(self.interval),
                "-s", "0",
                "-B", "8192",  # Increased buffer size
                "-n",
                "-U",  # Immediate write to file
                "-K",  # Disable packet checksum verification
                "-q",
                "-Z", "root"
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Wait for the duration of one capture interval
                time.sleep(self.interval)

                # If tcpdump exited prematurely, report the error
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    print(f"tcpdump terminated with error: {stderr.decode()}")
                    continue

                # Send SIGINT to stop the current tcpdump capture
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)

                # Report file size
                file_size = os.path.getsize(output_file)
                print(f"Saved traffic to {output_file} ({file_size} Bytes)")

            except subprocess.TimeoutExpired:
                print("Timeout while stopping tcpdump")
                process.kill()
            except Exception as e:
                print(f"Error during capturing: {e}")
                break

    def cleanup(self):
        """
        Stops any remaining tcpdump processes to keep the system clean.
        """
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'tcpdump':
                proc.kill()


def main():
    """
    Main function to instantiate the TrafficCapture class and start the capture.
    Adjust 'interval' and 'file_count' for your needs.
    """
    traffic_capture = TrafficCapture(
        traffic_type="normal_traffic_comparison",
        interval=3,  # Capture duration per file (in seconds)
        file_count=500,  # Number of .pcap files to create
        interface="wlp0s20f3"  # Replace with your actual interface
    )

    try:
        traffic_capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProgram is shutting down...")
    finally:
        traffic_capture.cleanup()


if __name__ == "__main__":
    main()

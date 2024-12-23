import subprocess
import os
import time
import signal
import psutil
from datetime import datetime


class ProtonTrafficCapture:
    def __init__(
            self,
            traffic_type="proton_vpn_traffic",
            interval=10,
            file_count=5,
            interface="tun0"
    ):
        """
        Initializes the ProtonTrafficCapture object for capturing ProtonVPN traffic.

        :param traffic_type: A label for the output folder (e.g., 'proton_vpn_traffic').
        :param interval: Duration (in seconds) for each tcpdump capture file.
        :param file_count: Number of capture files to create.
        :param interface: The network interface used by ProtonVPN (e.g., 'tun0').
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
        Ensures that the output directory exists.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _check_disk_space(self):
        """
        Checks available disk space in the output directory and warns if less than 1 GB is free.
        """
        disk_info = psutil.disk_usage(self.output_dir)
        free_space_gb = disk_info.free / (1024 * 1024 * 1024)
        if free_space_gb < 1:
            print(f"Warning: Only {free_space_gb:.2f} GB free disk space left.")
        return free_space_gb

    def capture_traffic(self):
        """
        Starts interval-based traffic capture on the ProtonVPN interface.
        Captures all traffic (no port filter) to .pcap files.
        """
        free_space_gb = self._check_disk_space()
        if free_space_gb < 1:
            print("Insufficient disk space. Aborting capture.")
            return

        for i in range(1, self.file_count + 1):
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            output_file = os.path.join(self.output_dir, f"capture_{i:03d}_{timestamp}.pcap")

            # No port filter here, so it captures all traffic on the ProtonVPN interface.
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

                time.sleep(self.interval)

                # If tcpdump exited prematurely, log the error
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    print(f"tcpdump terminated with error: {stderr.decode()}")
                    continue

                # Send SIGINT to end this interval capture cleanly
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)

                # Report file size
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"Saved traffic to {output_file} ({file_size} Bytes)")
                else:
                    print(f"Failed to create file: {output_file}")

            except subprocess.TimeoutExpired:
                print("Timeout while stopping tcpdump.")
                process.kill()
            except Exception as e:
                print(f"Error during capturing: {e}")
                break

    def cleanup(self):
        """
        Kills any lingering tcpdump processes to keep the system clean.
        """
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'tcpdump':
                proc.kill()
        print("Cleaned up lingering tcpdump processes.")


def main():
    """
    Main function to set up and start the ProtonVPN traffic capture.
    """
    # Adjust these parameters as needed
    capture = ProtonTrafficCapture(
        traffic_type="proton_vpn_capture_3_Sec_500",
        interval=3,  # Capture duration (seconds) per file
        file_count=500,  # Number of files to create
        interface="tun0"  # 'tun0' is the actual ProtonVPN interface
    )

    try:
        capture.capture_traffic()
    except KeyboardInterrupt:
        print("\nProton VPN traffic capture interrupted by user.")
    finally:
        capture.cleanup()


if __name__ == "__main__":
    main()

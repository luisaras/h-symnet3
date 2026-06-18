import sys, subprocess, os, signal, atexit, time

class RDDLServer:

	def __init__(self, prost_root="../../prost", domain_folder="", domain_root=None, episodes=30):
		self.prost_root = prost_root
		self.domain_folder = domain_folder
		self.domain_root = domain_root
		self.episodes = episodes
		self.proc = None
		self.register_global_cleanup()

	def start(self):
		root = self.prost_root + "/testbed"
		cmd = ["python3", root + "/run-server.py",
			"-r", str(self.episodes)]
		if self.domain_root:
			# Custom domain folder
			cmd.append("-b")
			folder = self.domain_root
			if self.domain_folder:
				folder += "/" + self.domain_folder
			cmd.append(folder)
		elif self.domain_folder:
			# Default benchmark folder
			cmd.append("-b")
			cmd.append(f"{root}/benchmarks/{self.domain_folder}")
		else:
			# Default benchmark folder (all domains)
			cmd.append("--all-ipc-benchmarks")
		print(cmd)
		try:
			self.proc = subprocess.Popen(cmd, 
				stdout=subprocess.PIPE, 
				stderr=subprocess.PIPE,
				preexec_fn=os.setsid,
				text=True)
			# Waiting until it's running
			for line in self.proc.stdout:
				if "RDDL Server Initialized" in line:
					print("RDDLSym Server Running ")
					return True
				print(line)
		except subprocess.CalledProcessError as e:
			print(f"Command failed with exit code {e.returncode}")
			print(e.stderr)
			return False


	def stop(self):
		if self.proc and self.proc.poll() is None:
			try:
				# Kill the whole process group (important!)
				os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
			except Exception:
				pass

	def __enter__(self):
		if not self.start():
			raise Exception("Couldn't connect to RDDLSim server.")
		return self

	def __exit__(self, exc_type, exc, tb):
		self.stop()

	def register_global_cleanup(self):
		def cleanup(*args):
			self.stop()
			sys.exit(1)

		# On normal exit
		atexit.register(self.stop)

		# On crash / Ctrl+C / kill
		signal.signal(signal.SIGINT, cleanup)
		signal.signal(signal.SIGTERM, cleanup)

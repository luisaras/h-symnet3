import sys, subprocess, os, signal, atexit, time

class RDDLServer:

	def __init__(self, prost_root="../../prost", domain_folder="", episodes=30):
		self.prost_root = prost_root
		self.domain_folder = domain_folder
		self.episodes = episodes
		self.proc = None
		self.register_global_cleanup()

	def start(self):
		root = self.prost_root + "/testbed"
		cmd = ["python3", root + "/run-server.py",
			"-r", str(self.episodes)]
		if self.domain_folder == "":
			cmd.append("--all-ipc-benchmarks")
		else:
			cmd.append("-b")
			cmd.append(f"{root}/benchmarks/{self.domain_folder}")
		print(cmd)
		self.proc = subprocess.Popen(cmd, 
			stdout=subprocess.PIPE, 
			stderr=subprocess.PIPE,
			preexec_fn=os.setsid,
			text=True)
		# Waiting until it's running
		for line in self.proc.stdout:
			if "RDDL Server Initialized" in line: break
			print(line)
		print("RDDLSym Server Running ")


	def stop(self):
		if self.proc and self.proc.poll() is None:
			try:
				# Kill the whole process group (important!)
				os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
			except Exception:
				pass

	def __enter__(self):
		self.start()
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

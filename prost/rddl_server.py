import sys, subprocess, os, signal, atexit, time

class RDDLServer:

	def __init__(self, prost_root="../../prost", benchmark="", episodes=30, port_shift=0):
		self.prost_root = prost_root
		self.benchmark = benchmark
		self.episodes = episodes
		self.proc = None
		self.port = str(2323 + port_shift)
		self.register_global_cleanup()

	def start(self):
		testbed = os.path.join(self.prost_root, "testbed")
		cmd = ["python3", os.path.join(testbed, "run-server.py"),
			"-r", str(self.episodes),
			"-p", self.port]
		if not self.benchmark:
			# Default benchmark folder (all domains)
			cmd.append("--all-ipc-benchmarks")
		else:
			cmd.append("-b")
			cmd.append(self.benchmark)
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
					print("RDDLSim Server Running on port " + self.port)
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
				print("Closed RDDLSim Server on port " + self.port)
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

import sys, subprocess, os, argparse

class Converter:

	def __init__(self, domain_folder, rddlsim_root="../../rddlsim"):
		self.domain_folder = domain_folder
		self.rddlsim_root = rddlsim_root

	def from_ppddl(self)
		rddl_folder = os.path.abspath(os.path.join(self.domain_folder, "domains"))
		ppddl_folder = os.path.abspath(os.path.join(self.domain_folder, "ppddl"))
		cmd = ["./run", "rddl.translate.RDDL2Format", rddl_folder, ppddl_folder, "ppddl"]
		try:
			process = subprocess.Popen(cmd,
				cwd=self.rddlsim_root,
				stdout=subprocess.PIPE,    # Capture stdout
				#stderr=subprocess.STDOUT,  # Redirect stderr into stdout so everything is in one place
				text=True,                 # Decode bytes to a string
				#check=True                 # Raise error on crash
			)
			output = ""
			for line in process.stdout:
				print(line, end="")        # Show on screen instantly
				output += "\n" + line  # Save to list
			process.wait()
			print(output)
			return True
		except subprocess.CalledProcessError as e:
			print(f"Command failed with exit code {e.returncode}")
			print(e.stdout)
			return False


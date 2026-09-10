import os, argparse

def create_dataset(domain, first_instance, last_instance, prost_log, save_folder):
	episodes = []
	transitions = []
	instances = [str(i) for i in range(first_instance, last_instance+1)]
	for i in instances:
		episodes = []
		f = open(os.path.join(prost_log, i+".result"))
		for line in f.readlines():
			
			if ">>> END OF ROUND" in line:
				episodes.append(transitions)
				transitions = []

			# Current state: | 1 0 1 0 0 0 0 1 1 
			if "Current state:" in line:
				res = line.split(":")[1].split("|")
				state = res[0].strip()+" "+res[1].strip()

			# Submitted action: set(x2, y2) 
			if "Submitted action:" in line:
				action = line.split(":")[1].strip().replace(" ", "")

			# Immediate reward: -1.000000
			if "Immediate reward:" in line:
				reward = line.split(":")[1].strip()
				transitions.append([i, state, action, reward])


		f = open(os.path.join(save_folder, i+".csv"), "w")

		for ep in episodes:
			for t in ep:
				res = str(t[0]) + ":"
				res += ",".join(t[1].strip().split(" ")) + ":"
				res += str(t[2]) + ":"
				res += str(t[3])
				f.write(res+"\n")

		f.close()
	print("Built dataset for domain " + domain)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("domain", help="name of the domain")
	parser.add_argument("-i", "--ins", help="first and last instances", 
		nargs=2, type=int, default=[1, 254])
	parser.add_argument("-l", "--prost_log", default=".",
		help="path of prost logs")
	parser.add_argument("-d", "--save_folder", help="folder to save dataset")
	args = parser.parse_args()

	if not os.path.isdir(args.save_folder):
		os.mkdir(args.save_folder)
	
	create_dataset(
		domain=args.domain, 
		first_instance=args.ins[0],
		last_instance=args.ins[1],
		prost_log=args.prost_log,
		save_folder=args.save_folder
	)
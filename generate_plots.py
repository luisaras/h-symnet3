#!/usr/bin/env python3
# =============================================================================

import os, argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'svg.fonttype': 'none'})  # keep text as text in SVG

METHODS = {"standard": "Baseline", "standard-lmc": "LM-Cut"}
DOMAINS = {"navigation": "Deterministic Navigation", "sysadmin": "SysAdmin"}

FOLDER_TEMPLATE = "{domain}_{method}"
LOSSES_TEMPLATE = "ckpt-{ckpt}_losses.csv"
REWARDS_TEMPLATE = "ckpt-{ckpt}_rewards.csv"

EPOCHS = 10 # number of epochs before each evaluation round

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Generate plots from model training logs",
		formatter_class=formatter,
	)
	default_root = os.path.join('.', 'multi_train', 'deep_plan', 'models')
	default_out_dir = os.path.join('.', 'plots')

	parser.add_argument(	     'domains',	help="domain name (navigation, sysadmin)",
		type=str, nargs="*")
	parser.add_argument('-n',	  '--ckpt',	help="model's checkpoint number",
		type=int, default=100)
	parser.add_argument('-r',	  '--root',	help='root folder containing the log folders',
		type=str, default=default_root)
	parser.add_argument('-o',  '--out_dir',	help='output directory for SVGs and CSV',
		type=str, default=default_out_dir)
	parser.add_argument('-f',   '--format',	help='image format (png, jpg, svg, pdf)',
		type=str, default="png")
	return parser.parse_args()


def generate_all_plots(root: str, out_dir: str, ckpt: int, domain: str, ext: str):
	methods = []
	for method in METHODS.keys():
		folder = os.path.join(root, FOLDER_TEMPLATE.format(domain=domain, method=method), 'checkpoints')
		episodes = []
		losses = []
		for i in range(1, ckpt+1):
			losses_path = os.path.join(folder, LOSSES_TEMPLATE.format(ckpt=i))
			rewards_path = os.path.join(folder, REWARDS_TEMPLATE.format(ckpt=i))
			df_episodes_i = pd.read_csv(rewards_path, delimiter="\t")
			df_episodes_i["epoch"] = i * EPOCHS
			df_losses_i = pd.read_csv(losses_path, delimiter="\t")
			episodes.append(df_episodes_i)
			losses.append(df_losses_i)

		df_episodes = pd.concat(episodes).sort_values("epoch")
		df_losses = pd.concat(losses).sort_values("epoch")

		df_avg_episodes = df_episodes.groupby("epoch").mean().reset_index()
		df_avg_losses = df_losses.groupby('epoch').mean().reset_index()

		plot_learning_curve(out_dir, df_avg_episodes, df_avg_losses, method, domain, ext)

		# Determine best reward
		best_avg_reward = float(df_avg_episodes["reward"].iloc[-1])

		# Determine convergence
		conv_epoch = estimate_convergence(df_episodes, reported_best_reward=best_avg_reward)

		# Total training time
		train_time = float(df_losses["time"].sum())

		# Average eval time
		eval_time = float(df_avg_episodes["time"].iloc[-1])

		methods.append({
			'method': method,
			'best_avg_reward': best_avg_reward,
			'converged_epoch': conv_epoch,
			'train_time': train_time,
			'eval_time': eval_time
		})

	df_methods = pd.DataFrame(methods)
	path = os.path.join(out_dir, f'{domain}.csv')
	df_methods.to_csv(path, index=False)
	print('\nSaved comparison table to', path)
	print('\nSummary:')
	print(df_methods.to_string(index=False))
	plot_comparison_bars(out_dir, df_methods, domain, ext)

	print('\nAll done.\n')


def plot_learning_curve(out_dir: str, df_avg_episodes, df_avg_losses, method: str, domain: str, ext: str):
	#ep_episodes = [df['reward'].mean() for df in episodes]
	#ep_epochs = range(1, (len(episodes)+1) * EPOCHS, EPOCHS)
	total_epochs = df_avg_losses["epoch"].max()

	# Axis x
	fig, ax = plt.subplots(figsize=(4,4))
	ax.set_xlabel('Epochs')
	ax.set_xlim(1, total_epochs)

	# Axis y: Reward
	color = "tab:blue"
	ax.set_ylabel('Average Total Reward', color=color)
	ax.tick_params(axis='y', labelcolor=color)
	ax.plot(df_avg_episodes["epoch"], df_avg_episodes["reward"], linewidth=1.5, color=color)
	
	# Axis y: Loss
	ax = ax.twinx() 
	color = "tab:red"
	#ax.set_yscale('log')
	ax.set_ylabel('Loss', color=color)
	ax.tick_params(axis='y', labelcolor=color)
	ax.plot(df_avg_losses["epoch"], df_avg_losses["loss"], alpha=0.5, linewidth=1, color=color)

	ax.set_title(f'{METHODS[method]} - {DOMAINS[domain]}')
	ax.grid(True, linestyle=':', linewidth=0.5)
	#ax.legend(loc='best')

	out_path = os.path.join(out_dir, f'{method}_{domain}.{ext}')
	fig.tight_layout()
	fig.savefig(out_path, format=ext)
	plt.close(fig)
	print("Generated learning curve: " + out_path)


def plot_comparison_bars(out_dir: str, df_methods, domain: str, ext: str):
	fig, ax = plt.subplots(figsize=(10,4))
	labels = df_methods["method"]
	values = df_methods["best_avg_reward"].fillna(0).astype(float)
	ax.bar(labels, values)
	ax.set_ylabel('Average reward (reported / observed max)')
	ax.set_title('Average reward by method')
	ax.set_xticklabels(labels, rotation=45, ha='right')

	out_path = os.path.join(out_dir, f'{domain}.{ext}')
	fig.tight_layout()
	fig.savefig(out_path, format=ext)
	plt.close(fig)


def rolling_mean(a, window):
	if len(a) == 0:
		return np.array([])
	if window <= 1:
		return np.array(a)
	return pd.Series(a).rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def estimate_convergence(df_episodes, reported_best_reward=None):
	"""
	Heuristic to estimate first episode where rolling mean (window=100 or smaller)
	reaches >= 95% of reported_best_reward (or 95% of observed max reward if None).
	Returns dict with episode index (1-based) and cumulative steps (sum of lengths) and
	rolling_mean array used.
	"""
	if df_episodes is None or len(df_episodes) == 0:
		return 0
	rewards = df_episodes['reward'].to_numpy()
	obs_max = float(np.nanmax(rewards))
	if reported_best_reward is None or np.isnan(reported_best_reward):
		target = 0.95 * obs_max
	else:
		target = 0.95 * float(reported_best_reward)
	win = min(100, max(1, len(rewards)//10))
	rm = rolling_mean(rewards, window=win)
	# find first index where rm >= target
	idx = None
	for i, v in enumerate(rm):
		if v >= target:
			idx = i  # 0-based
			break
	if idx is None:
		return 0
	return int(idx+1)


if __name__ == '__main__':
	args = parse_arguments()
	os.makedirs(args.out_dir, exist_ok=True)
	domains = args.domains if len(args.domains) > 0 else DOMAINS.keys()
	for domain in domains:
		generate_all_plots(args.root, args.out_dir, args.ckpt, domain, args.format)
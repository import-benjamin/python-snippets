#!/usr/bin/env python3

"""
This script require a unix shell with the following packages installed :
	- git
	- grep
	- sort
	- cut
"""

import subprocess
from pprint import pprint
from collections import Counter
from concurrent import futures


def get_contributions_author_list(file: str, condition: str = "author-mail") -> tuple:
	print(f"Scanning {file}")
	return (file, Counter([c for c in subprocess.run(f"git blame --line-porcelain HEAD {file} | grep \"^{condition} \" | sort | cut -d' ' -f2-", shell=True,  stdout=subprocess.PIPE, encoding="utf-8").stdout.split('\n') if c]))


def main():
	res = subprocess.run("git ls-tree -r -z --name-only HEAD", shell=True,  stdout=subprocess.PIPE, encoding="utf-8").stdout.split("\x00")
	res = [r for r in res if r] # clear empty item
	TF = 0

	with futures.ThreadPoolExecutor() as executor:
		result = executor.map(get_contributions_author_list, res)
	
	per_file_stats = dict(result)
	global_stats = sum(list(per_file_stats.values()), Counter())

	# pprint(per_file_stats)
	pprint(global_stats)
	
	tot_lines = sum(global_stats.values())
	authorship_percent = {key: global_stats[key]/tot_lines for key in global_stats}
	
	print(f"{' Current Authorship : ':=^80}")
	print(*[f"{key:<35} : {authorship_percent[key]*100:.02f}%" for key in authorship_percent], sep='\n')
	print(f"{' total : '+str(sum(authorship_percent.values())*100)+'% ':=^80}")

	

	while(sum(authorship_percent.values()) > 0.5):
		author = max(authorship_percent, key=authorship_percent.get)
		print(f"Critical author : {author}")
		authorship_percent.pop(author)
		TF += 1

	print(f"Estimated TruckFactor : {TF}")



if __name__ == '__main__':
	main()

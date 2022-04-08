# -*- coding:UTF-8 -*-
import os
import sys
from collections import Counter

def command(cmd): 
    res = os.popen(cmd)
    r = res.read().strip()
    res.close()
    return r

def run_spmv(): 

    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())

	for i in todo_mtx:
		print 'dealing with matrix ' + i
		r = command('ssget -ei' + str(mtx_2_id[i]))
		print 'executing matrix'
		command('CUDA_VISIBLE_DEVICES=1 ../spmv ' + r + '>> output.txt')
		print i + ' finished'

run_spmv()


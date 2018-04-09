import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import datetime
import itertools
import os
import time
import pytz
import cPickle as pickle
import tensorflow as tf

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_writer = None
_last_names = []

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	print_vals = [_iter[0]]

	names = list(_since_last_flush.keys())
	row_format_s = "{:^10}" + " | {:^16}" * len(names)
	row_format = "{:^10}" + " | {:^16.1e}" * len(names)

	if _last_names != names:
		headers = ["Iteration"] + names
		print(row_format_s.format(*headers))
		print("=" * (10 + 18*len(names)))
		_last_names[:] = names

	for name, vals in _since_last_flush.items():
		print_vals.append(np.mean(vals.values()))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		pltname = name.replace(' ', '_')+'.jpg'
		plt.savefig(os.path.join(default_folder(), default_experiment(), pltname))

	print(row_format.format(*print_vals))

	logfile = os.path.join(default_folder(), default_experiment(), 'log.pkl')
	with open(logfile, 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

	if _writer:
		for name, iter_dict in _since_last_flush.items():
			for i, val in iter_dict.items():
				summary = tf.Summary(
					value=[tf.Summary.Value(tag=name, simple_value=val)]
				)
				_writer.add_summary(summary, global_step=i)

	_since_last_flush.clear()


def set_writer(path):
  global _writer
  _writer = tf.summary.FileWriter(path)
  print('Writing to {}'.format(path))

START_TIME = datetime.datetime.now(pytz.timezone('US/Pacific'))


def start_time_str():
  return datetime.datetime.strftime(START_TIME, '%H%M%S')


def today_str():
  return datetime.datetime.strftime(START_TIME, '%Y-%m-%d')


def default_folder():
  return 'tmp_' + today_str()


def default_experiment():
  return 'tmp_' + start_time_str()


def default_tb_file():
  return start_time_str()


def full_tb_path(folder=None, experiment=None, tb_file=None):
  if folder is None:
    folder = default_folder()
  if experiment is None:
    experiment = default_experiment()
  if tb_file is None:
    tb_file = default_tb_file()

  return os.path.join(folder, experiment, tb_file)

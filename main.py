import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models import DrosophilaBrain, center_regions, left_right_regions

with open('Data/neuron.pkl', 'rb') as f:
  Neuron_dict = pickle.load(f)

with open('Data/conn.pkl', 'rb') as f:
  Connection_dict = pickle.load(f)


def run_step(i, x):
  spks, Vs = model.step_run(i, x)
  return spks, Vs


def run_simulation(model):
  indices = bm.arange(1000 / bm.dt)
  bg_inputs = bm.ones((len(indices), model.neuron_num)) * 0.1
  inputs = bm.zeros((len(indices), model.neuron_num))
  inputs[:, model.populations['LA']['slice']] = 2
  # inputs[:, :] = 20
  spks, Vs = bm.for_loop(run_step, [indices, inputs + bg_inputs], progress_bar=True, jit=True)

  firing_rate = []
  for region in center_regions + left_right_regions:
    region_fr = bp.measure.firing_rate(spks[:, model.populations[region]['slice']], dt=0.1, width=1000)
    firing_rate.append(region_fr.mean())

  plt.figure(figsize=(12, 6))  # 宽度为10英寸，高度为6英寸
  plt.plot(center_regions + left_right_regions, firing_rate, marker='o', markersize=4)
  plt.xticks(rotation=45)
  plt.xlabel('Brain Region')
  plt.ylabel('Firing Rate (Hz)')
  plt.tight_layout()
  plt.show()

  plt.plot(indices[:], Vs[:, model.populations['LOP']['slice']])
  plt.xlabel('Time (ms)')
  plt.ylabel('Membrane Potential (mV)')
  plt.show()

  # bp.visualize.raster_plot(indices[:], spks[:, model.populations['AL']['start']], show=True)


def plot_connection(model):
  conn = model.conn
  pre_ids, post_ids, weight = conn.require('pre_ids', 'post_ids', 'weight')
  # plot the img with pre, post and weight
  plt.figure(figsize=(12, 6))
  plt.scatter(pre_ids, post_ids, c=weight, s=1)
  plt.colorbar()
  plt.show()


if __name__ == '__main__':
  bm.set_platform('cpu')
  neuron_params = dict(V_rest=-52.,
                       V_reset=-52.,
                       V_th=-45.,
                       R=10.,
                       tau=20.,
                       tau_ref=2.2,
                       V_initializer=bp.init.OneInit(-52.),)

  syn_params = dict(tau=5., delay=1.8)

  model = DrosophilaBrain(neuron_dict=Neuron_dict, connection_dict=Connection_dict,
                          neuron_params=neuron_params, syn_params=syn_params)

  # plot_connection(model)
  run_simulation(model)

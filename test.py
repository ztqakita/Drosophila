import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models import DrosophilaBrain, center_regions, left_right_regions


def run_step(i, x):
  spks, Vs = model.step_run(i, x)
  return spks, Vs


def run_simulation(model):
  indices = bm.arange(1000 / bm.dt)
  bg_inputs = bm.ones((len(indices), model.neuron_num)) * 0.1
  inputs = bm.zeros((len(indices), model.neuron_num))
  inputs[:, model.populations['AL']['start']] = 0.7
  # inputs[:, :] = 20
  spks, Vs = bm.for_loop(run_step, [indices, inputs + bg_inputs], progress_bar=True, jit=False)

  firing_rate = []
  for region in center_regions + left_right_regions:
    region_fr = bp.measure.firing_rate(spks[:, model.populations[region]['slice']], dt=0.1, width=1000)
    firing_rate.append(region_fr.mean())

  plt.figure(figsize=(12, 6))  # 宽度为10英寸，高度为6英寸
  plt.plot(center_regions + left_right_regions, firing_rate, marker='o', markersize=4)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  plt.plot(indices[200:400], Vs[200:400, model.populations['AL']['slice']])
  plt.show()

  bp.visualize.raster_plot(indices[200:400], spks[200:400, model.populations['AL']['slice']], show=True)


if __name__ == '__main__':
  bm.set_platform('cpu')
  neuron_params = dict(V_rest=-52.,
                       V_reset=-52.,
                       V_th=-45.,
                       R=1.,
                       tau=20.,
                       tau_ref=2.2,
                       V_initializer=bp.init.OneInit(-52.), )

  model = bp.dyn.LifRef(size=10, **neuron_params)

  indices = bm.arange(1000 / bm.dt)
  inputs = bm.ones((len(indices), model.num)) * 0.8

  spks, Vs = bm.for_loop(run_step, [indices, inputs], progress_bar=True, jit=True)
  plt.plot(indices, Vs)
  plt.show()

  bp.visualize.raster_plot(indices, spks, show=True)

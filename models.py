import brainpy as bp
import brainpy.math as bm
import numpy as np
from customize import EventCSRWeight

non_conns = ['NO_CONS']

center_regions = ['NO', 'PB', 'EB', 'FB',   # Central Complex: Involved in multiple complex behaviors
                  'SAD', 'PRW',             # Periesophageal Neuropils Center: Sensory (including gustatory) processing,
                                            # motor control, and potentially circadian rhythm.
                  'GNG',                    # controlling the mouthparts, feeding behavior and taste info
                  'OCG',                    # Ocelli: contribute to light detection and circadian rhythm
                  'UNASGD']                 # Unassigned: no known function

left_right_regions = ['AME', 'LA', 'LO', 'LOP', 'ME',       # Optic lobe: Visual sensory region
                      'BU', 'LAL', 'GA',                    # Lateral Complex: Integration of sensory information
                      'LH',                                 # Lateral Horn: Olfactory sensory region
                      'CAN', 'AMMC', 'FLA',                 # Periesophageal Neuropils: Gustatory sensory region, motor control
                      'ICL', 'IB', 'ATL', 'CRE', 'SCL',     # Inferior Neuropils: Sensory integration, motor control
                      'VES', 'GOR', 'SPS', 'IPS', 'EPA',    # Ventromedial Neuropils: Sensory integration, learning and memory
                      'MB_PED', 'MB_VL', 'MB_ML', 'MB_CA',  # Mushroom Body: Olfactory learning and memory
                      'AL',                                 # Antennal Lobe: Olfactory sensory region
                      'SLP', 'SIP', 'SMP',                  # Superior Neuropils: Integration of multi-modality sensory information
                      'AVLP', 'PVLP', 'WED', 'PLP', 'AOTU'] # Ventrolateral Neuropils: Sensory processing and cognitive functions

Exc_nt_type = ['ACH', 'DA', 'OCT', 'SER']
Inh_nt_type = ['GABA', 'GLU']


class Exponential(bp.Projection):
  def __init__(self, pre, post, conn, syn_counts, pre_E, tau=5., delay=None):
    super().__init__()
    self.proj = bp.dyn.ProjAlignPreMg2(
      pre=pre,
      delay=delay,
      syn=bp.dyn.Expon.desc(size=post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
      comm=EventCSRWeight(conn=conn, syn_counts=syn_counts, pre_E=pre_E, post_V=post.V),
      out=bp.dyn.CUBA(),
      post=post
    )


class AlphaSyn(bp.Projection):
  def __init__(self, pre, post, conn, syn_counts, pre_E, tau=5., delay=None):
    super().__init__()
    self.proj = bp.dyn.ProjAlignPreMg2(
      pre=pre,
      delay=delay,
      syn=bp.dyn.Alpha.desc(size=post.num, tau_decay=tau, sharding=[bm.sharding.NEU_AXIS]),
      comm=EventCSRWeight(conn=conn, syn_counts=syn_counts, pre_E=pre_E, post_V=post.V),
      out=bp.dyn.CUBA(),
      post=post
    )


class DrosophilaBrain(bp.Network):
  def __init__(self, neuron_dict, connection_dict, neuron_params, syn_params):
    super(DrosophilaBrain, self).__init__()
    self.neuron_dict = neuron_dict
    self.connection_dict = connection_dict

    sum = 0
    for region in center_regions + left_right_regions:
      sum += self.neuron_dict[region]['neuron_number']
    self.neuron_num = sum

    self.populations = {}
    self.neu_pop = bp.dyn.LifRef(size=self.neuron_num,
                                 **neuron_params)

    start_index = 0
    neuron_index = []
    nt_type = []
    for region in center_regions + left_right_regions:
      end_index = start_index + self.neuron_dict[region]['neuron_number']
      self.populations[region] = {
        'slice': slice(start_index, end_index),
        'start': start_index,
        'end': end_index,
      }
      start_index = end_index
      neuron_index.append(self.neuron_dict[region]['neuron_ids'])
      nt_type.append(self.neuron_dict[region]['nt_type'])

    self.neuron_index = np.concatenate(neuron_index).reshape(-1)
    self.nt_type = np.concatenate(nt_type).reshape(-1)

    post_ids = []
    indptr = []
    syn_counts = []
    pre_E = []
    for region in center_regions + left_right_regions:
      r_post_index = self.connection_dict[region]['post_index']
      r_indptr = self.connection_dict[region]['indptr']
      r_syn_count = self.connection_dict[region]['syn_count']
      r_pre_nt_type = self.connection_dict[region]['syn_type']
      post_ids.append(r_post_index)
      indptr.append(r_indptr)

      # get pre neuron nt type
      # COBA version
      # r_pre_E = np.where(np.isin(r_pre_nt_type, Exc_nt_type), 0.0, -80.0)
      # CUBA version
      r_pre_E = np.where(np.isin(r_pre_nt_type, Exc_nt_type), 1, -1)
      syn_counts.append(r_syn_count)
      pre_E.append(r_pre_E)

    self.post_ids = np.concatenate(post_ids).reshape(-1)
    indptr = np.concatenate(indptr).reshape(-1)
    # Accumulate indptr and add 0 as the first element
    self.indptr = np.concatenate(([0], np.cumsum(indptr)))
    self.syn_counts = np.concatenate(syn_counts).reshape(-1)
    self.pre_E = np.concatenate(pre_E).reshape(-1)

    self.conn = bp.conn.CSRConn(indices=self.post_ids, inptr=self.indptr, pre=self.neu_pop.num, post=self.neu_pop.num)
    # self.proj = Exponential(pre=self.neu_pop,
    #                         post=self.neu_pop,
    #                         conn=self.conn,
    #                         syn_counts=self.syn_counts,
    #                         pre_E=self.pre_E,
    #                         **syn_params)
    self.proj = AlphaSyn(pre=self.neu_pop,
                         post=self.neu_pop,
                         conn=self.conn,
                         syn_counts=self.syn_counts,
                         pre_E=self.pre_E,
                         **syn_params)

    # Poisson noise
    # self.inp = bp.dyn.PoissonGroup(self.neuron_num, 0.1)

  def update(self, inp):
    self.proj()
    self.neu_pop(inp)
    return self.neu_pop.spike, self.neu_pop.V
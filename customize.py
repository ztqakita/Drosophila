import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np

from typing import Optional, Union, Callable

class EventCSRWeight(bp.dnn.Layer):
  r"""Synaptic matrix multiplication with event CSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
  :math:`M` the synaptic weight using a CSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: bp.conn.TwoEndConnector,
      syn_counts,
      pre_E,
      post_V,
      sharding=None,
      mode=None,
      name=None,
      transpose=True,
  ):
    super(EventCSRWeight, self).__init__(name=name, mode=mode)
    assert sharding is None, 'Currently this model does not support sharding.'
    self.conn = conn
    self.indices, self.indptr = self.conn.require('csr')
    self.sharding = sharding
    self.pre_E = pre_E
    self.post_V = post_V[self.indices]
    self.transpose = transpose

    # weight
    weight = bp.init.parameter(syn_counts * 0.275 / 10, (self.indices.size,))
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, x):
    # COBA version
    # weight = self.weight * (self.pre_E - self.post_V)
    # CUBA version
    weight = self.weight * self.pre_E

    if x.ndim == 1:
      return bm.event.csrmv(weight, self.indices, self.indptr, x,
                            shape=(self.conn.pre_num, self.conn.post_num),
                            transpose=self.transpose)
    elif x.ndim > 1:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_csrmv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_csrmv(self, x):
    return bm.event.csrmv(self.weight, self.indices, self.indptr, x,
                          shape=(self.conn.pre_num, self.conn.post_num),
                          transpose=self.transpose)
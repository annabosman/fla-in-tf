from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class ProgressiveGradientWalkOptimizer(optimizer.Optimizer):
    """Implementation of Progressive Gradient Walk as an Optimizer.
    """

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse resource apply updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Dense resource apply updates are not supported.")

    def __init__(self, step_size=0.5, use_locking=False, name="ProgressiveGradientWalkOptimizer"):
        super(ProgressiveGradientWalkOptimizer, self).__init__(use_locking, name)
        self._step = step_size

        # Tensor versions of the constructor arguments, created in _prepare().
        self._step_t = None

    def _prepare(self):
        self._step_t = ops.convert_to_tensor(self._step, name="max_step_size")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        step_t = math_ops.cast(self._step_t, var.dtype.base_dtype)
        #step_t = tf.Print(step_t, [step_t], message="step_t: ")
        #m = self.get_slot(var, "m")
        #m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        random_steps = tf.random_uniform(var.shape, 0, step_t)
        grad_sign = tf.sign(grad)
        mask = tf.where(tf.equal(grad_sign, 0), -1*tf.ones_like(grad_sign), grad_sign)

        var_update = state_ops.assign_sub(var, random_steps * mask)  # Update 'ref' by subtracting 'value

        return control_flow_ops.group(*[var_update, random_steps, grad_sign, mask])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
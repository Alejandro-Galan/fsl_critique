##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import unittest

import torch
import torch.nn as nn


class DistanceNetwork(nn.Module):
    """Compute support/query cosine-like similarities for Matching Networks.

    The original implementation iterated over the support sequence. This
    vectorized version preserves the same output and normalization behaviour:
    it normalizes only support embeddings, not the query embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, support_set, input_image):
        """Return similarities with shape ``[batch_size, sequence_length]``.

        Args:
            support_set: Tensor of shape ``[sequence_length, batch_size, embedding_dim]``.
            input_image: Tensor of shape ``[batch_size, embedding_dim]``.
        """
        eps = 1e-10
        if support_set.dim() != 3 or input_image.dim() != 2:
            raise ValueError(
                "Expected support_set [sequence_length, batch_size, embedding_dim] "
                "and input_image [batch_size, embedding_dim]."
            )
        if support_set.size(1) != input_image.size(0) or support_set.size(2) != input_image.size(1):
            raise ValueError("support_set and input_image dimensions are incompatible.")

        dot_product = (support_set * input_image.unsqueeze(0)).sum(dim=2)
        support_magnitude = support_set.pow(2).sum(dim=2).clamp(eps, float("inf")).rsqrt()
        return (dot_product * support_magnitude).t()


class DistanceNetworkTest(unittest.TestCase):
    def test_forward_shape(self):
        support = torch.ones(3, 2, 4)
        query = torch.ones(2, 4)
        output = DistanceNetwork()(support, query)
        self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == '__main__':
    unittest.main()

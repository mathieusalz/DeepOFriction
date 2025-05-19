# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import warnings

import torch
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.domain.constraint.continuous import DeepONetConstraint
from physicsnemo.sym.domain.validator.discrete import GridValidator
from physicsnemo.sym.dataset.discrete import DictGridDataset

from physicsnemo.sym.key import Key

import pandas as pd

from CVP_plotter import CustomValidatorPlotter
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.validator import PointwiseValidator

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # [datasets]
    # load training data
    file_path = "/home/salz/DeepONet/synthetic_data_generation/combined_dataset.csv"

    data = pd.read_csv(file_path)
    cols = len(data.columns) - 2

    branch_net_input_keys = [Key(data.columns[i]) for i in range(cols)]

    t_train = data["t"].to_numpy().reshape(-1,1)
    fric_train = data["friction_coefficient"].to_numpy().reshape(-1,1)
    
    # [init-model]
    # make list of nodes to unroll graph on
    trunk_net = FourierNetArch(
        input_keys=[Key("t")],
        output_keys=[Key("trunk", 128)],
    )

    branch_net = FullyConnectedArch(
        input_keys=branch_net_input_keys,
        output_keys=[Key("branch", 128)],
    )

    deeponet = DeepONetArch(
        output_keys=[Key("friction_coefficient")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )

    nodes = [deeponet.make_node("deepo")]
    # [init-model]

    # [constraint1]
    # make domain
    domain = Domain()

    invar = {"t": t_train}
    for i in range(cols):
        invar[data.columns[i]] = data[data.columns[i]].to_numpy().reshape(-1,1)

    # [constraint1]
    interior = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar= invar,
        outvar={"friction_coefficient": fric_train.reshape(-1,1)},
        batch_size=cfg.batch_size.train,
    )
    domain.add_constraint(interior, "Residual")

    invar_numpy = {"t": t_train[:6*250]}
    for i in range(cols):
        invar_numpy[data.columns[i]] = data[data.columns[i]].to_numpy().reshape(-1,1)[:6*250]

    validator = PointwiseValidator(nodes=nodes,
                                    invar=invar_numpy,
                                    true_outvar={"friction_coefficient": fric_train[:6*250]},
                                    plotter=CustomValidatorPlotter())

    domain.add_validator(validator, 'inferencer')

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()

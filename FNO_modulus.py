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
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint

from physicsnemo.sym.key import Key

import pandas as pd

from FNO_plotter import CustomValidatorPlotter
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter


@physicsnemo.sym.main(config_path="conf", config_name="config_fno")
def run(cfg: PhysicsNeMoConfig) -> None:

    # [datasets]
    # load training data
    input_keys = [Key('velocity')]
    vel_path = "/home/salz/DeepONet/synthetic_data_generation/normalized_velocity.csv"
    vel_data = pd.read_csv(vel_path, header=None)
    velocity = vel_data.to_numpy().reshape(-1, 1, 250)  # shape (1000, 1, 250)
    vel_train = velocity[:750]
    vel_test = velocity[750:]

    output_keys = [Key('friction_coefficient')]
    fric_path = "/home/salz/DeepONet/synthetic_data_generation/normalized_fric_coef.csv"
    fric_data = pd.read_csv(fric_path, header = None)
    fric = fric_data.to_numpy().reshape(-1, 1, 250)  # shape (1000, 1, 250)
    fric_train = fric[:750]
    fric_test = fric[750:]

    invar_train = {
        "velocity": vel_train
    }
    outvar_train = {
        "friction_coefficient": fric_train
    }

    invar_test = {
        "velocity": vel_test
    }

    outvar_test = {
        "friction_coefficient": fric_test
    }

    train_dataset = DictGridDataset(invar=invar_train, outvar=outvar_train)
    test_dataset = DictGridDataset(invar=invar_test, outvar=outvar_test)

    # [init-model]
    # make list of nodes to unroll graph on
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
    )
    nodes = [fno.make_node("fno")]
    # [init-model]

    # [constraint1]
    # make domain
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
    )
    domain.add_validator(val, "test")

    invar_numpy = {"velocity": vel_test[:100]}

    validator = PointwiseValidator(nodes=nodes,
                                    invar=invar_numpy,
                                    true_outvar={"friction_coefficient": fric_test[:100]},
                                    plotter=CustomValidatorPlotter())
    
    domain.add_validator(validator, 'inferencer')

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()

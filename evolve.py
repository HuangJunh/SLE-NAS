# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import math

def aconpso(particle, gbest, pbest, velocity, params):
    """
    pso for architecture evolution
    fixed-length PSO, use standard formula, but add a strided layer number constraint
    """
    particle_length = params['particle_length']
    max_output_channel = params['max_output_channel']
    cur_len = len(particle)
    # 1.velocity calculation
    w, c1, c2 = 0.7298, 1.49618, 1.49618
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    new_velocity = np.asarray(velocity) * w + c1 * r1 * (np.asarray(pbest) - np.asarray(particle)) + c2 * r2 * (np.asarray(gbest) - np.asarray(particle))

    # 2.particle updating
    new_particle = list(particle + new_velocity)
    new_particle = [round(par, 2) for par in new_particle ]#particle里面的数必须为两位小数
    new_velocity = list(new_velocity)


    # 3.adjust the value according to some constraints
    subparticle_length = particle_length // 3
    subParticles = [new_particle[0:subparticle_length], new_particle[subparticle_length:2 * subparticle_length],
                    new_particle[2 * subparticle_length:]]

    for j, subParticle in enumerate(subParticles):
        valid_particle = [dimen for dimen in subParticle if 0 <= dimen <= 12.99]
        # condition 1：the number of valid layer (non-strided or strided layer, not identity) must >0
        if len(valid_particle) == 0:
            # if the updated particle has no valid value, let the first dimension value to 0.03 (3*3 DW-sep conv, no.filter=3)
            new_particle[j * subparticle_length] = 0.00

    # 4.outlier handling - maintain the particle and velocity within their valid ranges
    updated_particle1 = []
    for k,par in enumerate(new_particle):
        if (0.00 <= par <= 12.99):
            updated_particle1.append(par)
        elif par > 12.99:
            updated_particle1.append(12.99)
        else:
            updated_particle1.append(0.00)

    updated_particle = []
    for k, par in enumerate(updated_particle1):
        if int(round(par - int(par), 2) * 100) + 1  > max_output_channel:
            updated_particle.append(round(int(par) + float(max_output_channel-1)/100,2))
        else:
            updated_particle.append(par)

    return updated_particle, new_velocity


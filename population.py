# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
    population initialization
"""
import numpy as np

def initialize_population(params):
    pop_size = params['pop_size']
    particle_length = params['particle_length']
    max_output_channel = params['max_output_channel']
    population = []
    for _ in range(pop_size):
        particle = []

        num_net = int(particle_length)  # fixed particle length

        num_identity = np.random.randint(0, int(num_net*2/3))
        dimens_identity = [0]*num_identity
        dimens = list(np.random.randint(1, 13, size=num_net-num_identity))
        dimens.extend(dimens_identity)
        np.random.shuffle(dimens)
        
        for i in range(num_net):
            dimen_ll = np.random.randint(0, int(max_output_channel*2/3))   # conv (stride=1)
            particle.append(round(dimens[i] + float(dimen_ll)/100,2))

        subparticle_length = num_net // 3
        subParticles = [particle[0:subparticle_length], particle[subparticle_length:2 * subparticle_length], particle[2 * subparticle_length:]]
        for j, subParticle in enumerate(subParticles):
            valid_particle = [dimen for dimen in subParticle if 0 <= dimen <= 12.99]
            # condition 1：the number of valid layer (non-strided or strided layer, not identity) must >0
            if len(valid_particle) == 0:
                # if the updated particle has no valid value, let the first dimension value to 0.03 (3*3 DW-sep conv, no.filter=3)
                particle[j * subparticle_length] = 0.00
        
        population.append(particle)
    return population

def test_population():
    params = {}
    params['pop_size'] = 20
    params['particle_length'] = 24
    params['max_output_channel'] = 100

    pop = initialize_population(params)
    print(pop)

if __name__ == '__main__':
    test_population()

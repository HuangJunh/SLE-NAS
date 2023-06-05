# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log, GPUTools
from multiprocessing import Process
import importlib
import sys,os, time
import numpy as np
import copy
from asyncio.tasks import sleep

def decode(particle, curr_gen, id):

    pytorch_filename = Utils.generate_pytorch_file(particle, curr_gen, id)

    return pytorch_filename

def check_all_finished(filenames, curr_gen):
    filenames_ = copy.deepcopy(filenames)
    output_file = './populations/err_%02d.txt' % (curr_gen)
    if os.path.exists(output_file):
        f = open(output_file, 'r')
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                if line[0] in filenames_:
                    filenames_.remove(line[0])
        f.close()
        if filenames_:
            return False
        else:
            return True
    else:
        return False

def fitnessEvaluate(filenames, curr_gen, is_test, population, batch_size_set=None, weight_decay_set=None):
    err_params_flops_set = np.zeros(shape=(3, len(filenames)))
    has_evaluated_offspring = False
    p = None
    for i, file_name in enumerate(filenames):
        has_evaluated_offspring = True
        # time.sleep(40)
        if p:
            p.join()
        gpu_id = GPUTools.detect_available_gpu_id()
        while gpu_id is None:
            time.sleep(60)
            gpu_id = GPUTools.detect_available_gpu_id()
        if gpu_id is not None:
            Log.info('Begin to train %s' % (file_name))
            module_name = 'scripts.%s' % (file_name)
            if module_name in sys.modules.keys():
                Log.info('Module:%s has been loaded, delete it' % (module_name))
                del sys.modules[module_name]
                _module = importlib.import_module('.', module_name)
            else:
                _module = importlib.import_module('.', module_name)
            _class = getattr(_module, 'RunModel')
            cls_obj = _class()
            if batch_size_set:
                p = Process(target=cls_obj.do_work, args=(
                '%d' % (gpu_id), curr_gen, file_name, is_test, population[i], batch_size_set[i], weight_decay_set[i]))
            else:
                p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), curr_gen, file_name, is_test, population[i]))
            p.start()

    p.join()
    time.sleep(10)

    if has_evaluated_offspring:
        file_names = ['./populations/err_%02d.txt' % (curr_gen), './populations/params_%02d.txt' % (curr_gen), './populations/flops_%02d.txt' % (curr_gen)]
        fitness_maps = [{},{},{}]
        for j, file_name in enumerate(file_names):
            assert os.path.exists(file_name) == True
            f = open(file_name, 'r')

            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_maps[j][line[0]] = float(line[1])
            f.close()

            for i in range(len(err_params_flops_set[0])):
                if filenames[i] not in fitness_maps[j]:
                    Log.warn(
                        'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 60 seconds' % (
                            filenames[i], file_name))
                    sleep(120)
                err_params_flops_set[j][i] = fitness_maps[j][filenames[i]]

    else:
        Log.info('None offspring has been evaluated')

    return list(err_params_flops_set[0]), list(err_params_flops_set[1]), list(err_params_flops_set[2])






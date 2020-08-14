# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import numpy as np
import cntk
import _cntk_py
import cntk.io.transforms
import cntk.io
import os.path
import time
from flattenandCategorize import writeCategories
from cntk import Trainer, cntk_py
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.debugging import set_computation_network_trace_level
from cntk.logging import *
from cntk.debugging import *
from ResNetModels import *
# Paths relative to current python file.
abs_path   = "E:/img/Animals/"#os.path.dirname(os.path.abspath(__file__))
data_path  = (abs_path)
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 100
image_width  = 100
num_channels = 3  # RGB
num_classes  = 4

# Define the reader for both training and evaluation action.
def create_reader(map_file, is_training):
    if not os.path.exists(map_file) :
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms1 = []
    if is_training:
        transforms1 += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]
    transforms1 += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # deserializer
    return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='image', transforms=transforms1), # first column in map file is referred to as 'image'
        labels   = cntk.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)

# Train and evaluate the network.
def convnet_cifar10_dataaug(reader_train, reader_test, epoch_size = 176318, max_epochs = 1000):
    _cntk_py.set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = cntk.ops.input_variable((num_channels, image_height, image_width))
    label_var = cntk.ops.input_variable((num_classes))

    # apply model to input
    scaled_input = cntk.ops.element_times(cntk.ops.constant(0.00390625), input_var)

    z = create_cifar10_model(input_var, 6, num_classes)
    lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]

    # loss and metric
    ce = cntk.losses.cross_entropy_with_softmax(z, label_var)
    pe = cntk.metrics.classification_error(z, label_var)

    # training config
    minibatch_size = 100

    # Set learning parameters
    lr_per_sample          = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    lr_schedule            = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learners.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant       = [0]*20 + [600]*20 + [1200]
    mm_schedule            = cntk.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight          = 0.002
    progress_printer = cntk.logging.ProgressPrinter(tag='Training')
    # trainer object
    learner = cntk.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                                        l2_regularization_weight = l2_reg_weight)
    trainer =  cntk.Trainer(z, (ce, pe), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    cntk.logging.log_number_of_parameters(z) ; print()
    
    ModelFile = os.path.join(model_path, "ResNet_PredatorDeep2.dnn")
    if (os.path.isfile(ModelFile)):
        print ("restored from backup")
        trainer.restore_from_checkpoint(ModelFile)
    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

        print (epoch, max_epochs)
        progress_printer.epoch_summary(with_metric=True)
        trainer.save_checkpoint(ModelFile)
        time.sleep(2)
        z.save_model(os.path.join(model_path, "ResNet_PredatorDeepmodel2.dnn"))
        
        trainer.save_checkpoint(os.path.join(model_path, "ResNet_PredatorDeepBack2.dnn"))
  

if __name__=='__main__':
    Paths = ["E:/img/Animals/Feline/","E:/img/Animals/Canine/","E:/img/Animals/FlyingPred/","E:/img/Animals/OK/"]
    epochSize = writeCategories(os.path.join(data_path, 'AnimalsFullSize.txt'),Paths)
    print ("epoch size",epochSize)
    reader_train = create_reader(os.path.join(data_path, 'AnimalsFullSize.txt'), True)
    reader_test  = create_reader(os.path.join(data_path, 'AnimalsFullSize.txt'), False)

    convnet_cifar10_dataaug(reader_train, reader_test,epochSize,20000)


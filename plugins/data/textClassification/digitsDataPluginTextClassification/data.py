# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import csv
import os
import random

import numpy as np

from digits.utils import subclass, override, constants
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm


DATASET_TEMPLATE = "templates/dataset_template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for text classification
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        self.userdata['cdict'] = {c: i + 1 for i, c in enumerate(self.alphabet)}
        # assign unknown characters to the same next available ID
        self.userdata['unknown_char_id'] = len(self.alphabet) + 1

        if self.class_labels_file:
            with open(self.class_labels_file) as f:
                self.userdata['class_labels'] = f.read().splitlines()

    @override
    def encode_entry(self, entry):
        label = np.array([int(entry['class'])])

        # convert characters to numbers
        sample = []
        count = 0
        max_chars = self.max_chars_per_sample
        for field in entry['fields']:
            for char in field.lower():
                if not max_chars or count >= self.max_chars_per_sample:
                    break
                num = (
                    self.userdata['cdict'][char]
                    if char in self.userdata['cdict']
                    else self.userdata['unknown_char_id']
                )
                sample.append(num)
                count += 1
        # convert to numpy array
        sample = np.array(sample, dtype='uint8')
        # pad if necessary
        if max_chars and count < max_chars:
            sample = np.append(sample, np.full(
                (max_chars - count),
                fill_value=self.userdata['unknown_char_id'],
                dtype='uint8'))
        # make it a 3-D array
        sample = sample[np.newaxis, np.newaxis, :]
        return sample, label

    @staticmethod
    @override
    def get_category():
        return "Text"

    @staticmethod
    @override
    def get_id():
        return "text-classification"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated
           with values if the job was cloned
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, DATASET_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_inference_form():
        return InferenceForm()

    @staticmethod
    @override
    def get_inference_template(form):
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, INFERENCE_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "Classification"

    @override
    def itemize_entries(self, stage):
        if self.userdata['is_inference_db']:
            if stage == constants.TEST_DB:
                if not (bool(self.test_data_file) ^ bool(self.snippet)):
                    raise ValueError("You must provide either a data file or a snippet")
                if self.test_data_file:
                    entries = self.read_csv(self.test_data_file, False)
                elif self.snippet:
                    entries = [{'class': '0', 'fields': [self.snippet]}]
            else:
                entries = []
        elif (
            stage != constants.TRAIN_DB
            and stage == constants.VAL_DB
            and self.val_data_file
        ):
            entries = self.read_csv(self.val_data_file)
        elif (
            stage != constants.TRAIN_DB
            and stage == constants.VAL_DB
            or stage != constants.TRAIN_DB
        ):
            entries = []
        else:
            entries = self.read_csv(self.train_data_file)
        return entries

    def read_csv(self, filename, shuffle=True):
        entries = []
        with open(filename) as f:
            reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
            entries.extend(iter(reader))
        random.shuffle(entries)
        return entries

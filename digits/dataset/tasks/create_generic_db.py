# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import sys

import digits
from digits.task import Task
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled version changes
PICKLE_VERSION = 1


@subclass
class CreateGenericDbTask(Task):
    """
    A task to create a db using a user-defined extension
    """

    def __init__(self, job, backend, stage, **kwargs):
        """
        Arguments:
        """
        self.job = job
        self.backend = backend
        self.stage = stage
        self.create_db_log_file = f"create_{stage}_db.log"
        self.dbs = {'features': None, 'labels': None}
        self.entry_count = 0
        self.feature_shape = None
        self.label_shape = None
        self.mean_file = None
        super(CreateGenericDbTask, self).__init__(**kwargs)
        self.pickver_task_create_generic_db = PICKLE_VERSION

    @override
    def name(self):
        return f'Create {self.stage} DB'

    @override
    def __getstate__(self):
        state = super(CreateGenericDbTask, self).__getstate__()
        if 'create_db_log' in state:
            # don't save file handle
            del state['create_db_log']
        return state

    @override
    def __setstate__(self, state):
        super(CreateGenericDbTask, self).__setstate__(state)
        self.pickver_task_create_generic_db = PICKLE_VERSION

    @override
    def before_run(self):
        super(CreateGenericDbTask, self).before_run()
        # create log file
        self.create_db_log = open(self.path(self.create_db_log_file), 'a')
        # save job before spawning sub-process
        self.job.save()

    def get_encoding(self, name):
        if name == 'features':
            return self.job.feature_encoding
        elif name == 'labels':
            return self.job.label_encoding
        else:
            raise ValueError(f"Unknown db: {name}")

    @override
    def process_output(self, line):
        self.create_db_log.write('%s\n' % line)
        self.create_db_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        if match := re.match(
            f'Created (features|labels) db for stage {self.stage} in (.*)', message
        ):
            db_type = match.group(1)
            self.dbs[db_type] = match.group(2)
            return True

        if match := re.match(
            f'Created mean file for stage {self.stage} in (.*)', message
        ):
            self.mean_file = match.group(1)
            return True

        if match := re.match(
            r'Found (\d+) entries for stage %s' % self.stage, message
        ):
            count = int(match.group(1))
            self.entry_count = count
            return True

        if match := re.match(
            f'Feature shape for stage {self.stage}: (.*)', message
        ):
            self.feature_shape = eval(match.group(1))
            return True

        if match := re.match(f'Label shape for stage {self.stage}: (.*)', message):
            self.label_shape = eval(match.group(1))
            return True

        if match := re.match(r'Processed (\d+)\/(\d+)', message):
            self.progress = float(match.group(1)) / int(match.group(2))
            self.emit_progress_update()
            return True

        # errors, warnings
        if level == 'warning':
            self.logger.warning(f'{self.name()}: {message}')
            return True
        if level in ['error', 'critical']:
            self.logger.error(f'{self.name()}: {message}')
            self.exception = message
            return True

        return False

    @override
    def after_run(self):
        super(CreateGenericDbTask, self).after_run()
        self.create_db_log.close()

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from create_db_task_pool
        cpu_key = 'create_db_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                return reserved_resources
        return None

    @override
    def task_arguments(self, resources, env):
        return [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(digits.__file__)),
                'tools',
                'create_generic_db.py',
            ),
            self.job.id(),
            f'--stage={self.stage}',
            f"--jobs_dir={digits.config.config_value('jobs_dir')}",
        ]

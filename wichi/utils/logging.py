class Logger(object):
    def __init__(self):
        self.logged_objects = dict()

    def log_scalar(self, name, value, step=None):
        if name not in self.logged_objects:
            self.logged_objects[name] = list()
        self.logged_objects[name].append({'value': value, 'step': step})

    def get_log(self, name):
        return self.logged_objects[name]
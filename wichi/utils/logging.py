import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self):
        self.logged_objects = dict()
        self.axes = dict()

    def log_scalar(self, name, value, step=None):
        if name not in self.logged_objects:
            self.logged_objects[name] = list()
        self.logged_objects[name].append({'value': value, 'step': step})

    def get_log(self, name):
        return self.logged_objects[name]

    def plot(self, names, title):
        colors = ['green', 'orange', 'cyan', 'blue', 'magenta', 'red', 'black']
        if title not in self.axes:
            fig, ax = plt.subplots()
            self.axes[title] = ax
        for idx, name in enumerate(names):
            steps = [t['step'] for t in self.get_log(name)]
            losses = [t['value'] for t in self.get_log(name)]
            self.axes[title].plot(steps, losses, color=colors[idx % len(colors)])
        self.axes[title].set_title(title)
        self.axes[title].set_xlabel('step')
        self.axes[title].legend(names)
        plt.draw()
        plt.pause(0.5)

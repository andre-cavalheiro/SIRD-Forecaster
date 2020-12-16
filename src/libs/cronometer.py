import time

class cronometer():

    def __init__(self, rootName=''):
        self.name = rootName
        self.timer = {}

    def start(self, name):
        self.timer[name] = {
            'start': time.time()
        }

    def stop(self, name):
        assert(name in self.timer.keys())
        self.timer[name]['duration'] = time.time() - self.timer[name]['start']

    def printAll(self):
        if self.name:
            print('== Times for {} =='.format(self.name))
        else:
            print('== Times ==')
        for k, v in self.timer.items():
            print('- \"{}\" => {} seconds'.format(k, v['duration']))
        print('Total execution time {}'.format(sum([t['duration'] for t in self.timer.values()])))

# (c) 2007 The Board of Trustees of the University of Illinois.

class Future:
    def __init__(self, thunk):
        self.evaluated = 0
        self.value = thunk

    def get(self):
        if self.evaluated:
            return self.value
        else:
            self.value = self.value()
            self.evaluated = 1
            return self.value

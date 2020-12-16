

class NegativeActiveCases(Exception):
    def __init__(self, region, arr):
        self.region = region
        self.arr = arr
        super().__init__(f'Negative active cases in {region}\n {self.arr}')


class ActiveCasesWithNaN(Exception):
    def __init__(self, region, arr):
        self.region = region
        self.arr = arr
        super().__init__(f'Active cases with NaN in {region}\n {self.arr}')


class RecoveriesWithNaN(Exception):
    def __init__(self, region, arr):
        self.region = region
        self.arr = arr
        super().__init__(f'Recoveries with NaN in {region}\n {self.arr}')


class DeathsWithNaN(Exception):
    def __init__(self, region, arr):
        self.region = region
        self.arr = arr
        super().__init__(f'Deaths with NaN in {region}\n {self.arr}')

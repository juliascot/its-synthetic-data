'''
Tensor class. This will take in the csv file and desired rank of the tensor and process it
'''

class Tensor():
    def __init__(file: str, rank: int, if_stratify: bool = False) -> None: 

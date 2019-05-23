from .parallel_apply import parallel_apply
from .replicate import replicate
from .data_parallel import DataParallel, data_parallel
from .data_singular import DataSingular, data_singular

from .scatter_gather import scatter, gather

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel',
           'DataParallel', 'data_singular', 'DataSingular']

# __all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_singular', 'DataSingular']

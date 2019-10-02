import torch
from torch.nn import Module
from torch.autograd import Variable
from .scatter_gather import scatter_kwargs, gather, scatter
from .replicate import replicate
from .parallel_apply import parallel_apply


class DataSingular(Module):
    r"""Implements a module meant to only run a single implementation (not parallel)



    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    .. warning::
        Forward and backwrad hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    Args:
        module: module to be run a single time
        device_id: CUDA device_id or 'cpu'
        output_device: device location of output (default: device_ids[0])
        cpu_keywords: list of argument keywords that could be used in `forward` to
            indicating not moving the argument to gpu. Currently, only support
            argument of type: Variable

    Example::

        >>> net = torch.nn.DataSingular(model, device_ids=[0])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_id=None, output_device=None, dim=0,
                 cpu_keywords=[], minibatch=False, batch_outputs=True): #cfg_device=None
        super(DataSingular, self).__init__()
        # Orig Code:
        if not torch.cuda.is_available() and device_id != 'cpu':
            self.module = module
            self.device_id = []
            return

        if device_id is None:
            device_id = [0]
        if output_device is None:
            output_device = device_id[0]
        self.dim = dim
        self.module = module
        self.device_id = device_id
        self.output_device = output_device
        if len(self.device_id) == 1:
            self.module.to(torch.device(device_id[0]))
        self.cpu_keywords = cpu_keywords
        self.minibatch = minibatch
        self.batch_outputs = batch_outputs
        print(self.device_id , "Dataparallel device_id1")
        # self.device_id = ['cpu']




    def forward(self, *inputs, **kwargs):
        if not self.device_id:
            return self.module(*inputs, **kwargs)
        mini_inputs = [x[0] for x in inputs]
        mini_kwargs = dict([(k, v[0]) for k, v in kwargs.items()])


        # a, b = self._minibatch_scatter(self.device_id[0], *mini_inputs, **mini_kwargs)
        # //////////////////////////////////
        # "minibatch scatter function start"
        kwargs_cpu = {}
        for k in mini_kwargs:
            if k in self.cpu_keywords:
                kwargs_cpu[k] = mini_kwargs[k]
        for k in self.cpu_keywords:
            mini_kwargs.pop(k, None)
        # //////////////////////////////////
        # "self.scatter function start"

        # mini_inputs, mini_kwargs = self.scatter(mini_inputs, mini_kwargs, [self.device_id[0]])
                                        # scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

        # //////////////////////////////////
        # "scatter function start"
        # print()
        # print("type(mini_inputs)" , type(mini_inputs))
        # print("type(mini_kwargs)" , type(mini_kwargs))
        #
        # print("mini_inputs")
        # for i, el in enumerate(mini_inputs):
        #     print(i," ",el)
        # print()
        # print("mini_kwargs")
        # for k,v in mini_kwargs.items():
        #     print()
        #     print("key: ", k)
        #     print("value type", type(v))
        #     if torch.is_tensor(v):
        #         print(v.is_cuda, k, " is cuda")
        #         if v.is_cuda:
        #             print(torch.get_device(v))
        # mini_inputs = scatter(mini_inputs, [self.device_id[0]], self.dim) if mini_inputs else []
        # mini_kwargs = scatter(mini_kwargs, [self.device_id[0]], self.dim) if mini_kwargs else []
        mini_inputs = [mini_inputs]
        mini_kwargs = [mini_kwargs]
        for el in mini_kwargs:
            for k,v in el.items():
                if torch.is_tensor(v):
                    # print("moving to cuda0")
                    el[k] = v.to(torch.device(self.device_id[0]))
                    # print(torch.get_device(el[k]))

        # print("------------------------------------")
        # print("type(mini_inputs)" , type(mini_inputs))
        # print("type(mini_kwargs)" , type(mini_kwargs))
        #
        #
        # print("mini_kwargs")
        # for k,v in mini_kwargs[0].items():
        #     print()
        #     print("key: ", k)
        #     print("value type", type(v))
        #     if torch.is_tensor(v):
        #         print(v.is_cuda, k, " is cuda")
        #         if v.is_cuda:
        #             print(torch.get_device(v))

        # "scatter function end"
        # //////////////////////////////////

        if len(mini_inputs) < len(mini_kwargs):
            mini_inputs.extend([() for _ in range(len(mini_kwargs) - len(mini_inputs))])
        elif len(mini_kwargs) < len(mini_inputs):
            mini_kwargs.extend([{} for _ in range(len(mini_inputs) - len(mini_kwargs))])
        mini_inputs = tuple(mini_inputs)
        mini_kwargs = tuple(mini_kwargs)

        # "self.scatter function end"
        # //////////////////////////////////
        kwargs_cpu = [kwargs_cpu] # a list of dict
        # Merge cpu kwargs with gpu kwargs
        for d_gpu, d_cpu in zip(mini_kwargs, kwargs_cpu):
            d_gpu.update(d_cpu)
        a= mini_inputs[0]
        b= mini_kwargs[0]
        # "minibatch scatter function end"
        # //////////////////////////////////


        inputs_list = [a]
        kwargs_list = [b]





        inputs = inputs_list
        kwargs = kwargs_list


        outputs = [self.module(*inputs[0], **kwargs[0])]

#  old code
        # if len(self.device_id) == 1:
        #     outputs = [self.module(*inputs[0], **kwargs[0])]
        # else:
        #     replicas = self.replicate(self.module, self.device_id[:len(inputs)])
        #     outputs = self.parallel_apply(replicas, inputs, kwargs)
        # print()
        # print("len(outputs)", len(outputs))
        # print("type(outputs[0])", type(outputs[0]))
        # for k,v in outputs[0].items():
        #     print()
        #     print("key", k)
        #     print("type(v)", type(v))
        #     if isinstance(v, dict):
        #         print()
        #         for k2,v2 in v.items():
        #             print("     key", k2)
        #             print("     type(v)", type(v2))
        #             if (torch.is_tensor(v2)):
        #                 if not v2.is_cuda:
        #                     print("             CPU")
        #                 else:
        #                     print("            ", torch.get_device(v2))
        # outstuff = self.gather(outputs, 0)
        outstuff = outputs[0]
        # print()
        # print("len(outstuff)", len(outstuff))
        # print("type(outstuff)", type(outstuff))
        # for k,v in outstuff.items():
        #     print()
        #     print("key", k)
        #     print("type(v)", type(v))
        #     if isinstance(v, dict):
        #         print()
        #         for k2,v2 in v.items():
        #             print("     key", k2)
        #             print("     type(v)", type(v2))
        #             if (torch.is_tensor(v2)):
        #                 if not v2.is_cuda:
        #                     print("             CPU")
        #                 else:
        #                     print("            ", torch.get_device(v2))
        if ("rois" in outstuff.keys()):
            outstuff["rois"] = torch.from_numpy(outstuff["rois"])
        return outstuff


#  old code
        # if self.batch_outputs:
        #     return self.gather(outputs, self.output_device)
        # else:
        #     return [self.gather([x], self.output_device) for x in outputs]

    def _minibatch_scatter(self, device_id, *inputs, **kwargs):
        kwargs_cpu = {}
        for k in kwargs:
            if k in self.cpu_keywords:
                kwargs_cpu[k] = kwargs[k]
        for k in self.cpu_keywords:
            kwargs.pop(k, None)
        inputs, kwargs = self.scatter(inputs, kwargs, [device_id])
        kwargs_cpu = [kwargs_cpu] # a list of dict
        # Merge cpu kwargs with gpu kwargs
        for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
            d_gpu.update(d_cpu)
        return inputs[0], kwargs[0]

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_id[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

# From What I understand this functional version isn't used
def data_singular(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

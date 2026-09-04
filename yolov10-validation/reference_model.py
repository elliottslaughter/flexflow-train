"""Build the upstream (ultralytics) YOLOv10x model and describe how its tensors
correspond to FlexFlow's.

FlexFlow's ``lib/models/src/models/yolov10/yolov10.cc`` names each layer it
creates after the corresponding ultralytics module path (e.g.
``model.0.conv``), and ``ComputationGraphBuilder::add_layer`` names each weight
after the layer that consumes it plus the slot it is consumed in (e.g.
``model.0.conv.FILTER``).  That makes the correspondence between the two
frameworks' tensors purely mechanical, which is what the helpers here rely on.
"""

import torch
from torch import nn

from ultralytics.nn.modules.block import C2f, PSA, SCDown, SPPF
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.tasks import DetectionModel

# FlexFlow tensor slot name -> PyTorch parameter suffix.
SLOT_TO_PARAM = {
    "FILTER": "weight",  # conv2d kernel
    "BIAS": "bias",  # conv2d bias
    "GAMMA": "weight",  # batch norm scale
    "BETA": "bias",  # batch norm shift
}


def build_model(cfg="yolov10x.yaml", nc=80, seed=0):
    """Build a randomly-initialized ultralytics YOLOv10x model."""
    torch.manual_seed(seed)
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
    return model


def flexflow_weight_name_to_param_name(ff_name):
    """``model.0.conv.FILTER`` -> ``model.0.conv.weight``."""
    base, _, slot = ff_name.rpartition(".")
    if slot not in SLOT_TO_PARAM:
        raise ValueError(f"unrecognized weight slot in {ff_name!r}")
    return f"{base}.{SLOT_TO_PARAM[slot]}"


def get_weight_tensors(model, ff_weight_names):
    """Return a dict of FlexFlow weight name -> the matching PyTorch parameter."""
    params = dict(model.named_parameters())
    result = {}
    for ff_name in ff_weight_names:
        param_name = flexflow_weight_name_to_param_name(ff_name)
        if param_name not in params:
            raise KeyError(f"{ff_name} -> {param_name} not found in the model")
        result[ff_name] = params[param_name]
    return result


def get_module_output_names(model):
    """Map each PyTorch module path to the FlexFlow tensor that holds its output.

    An ultralytics ``Conv`` is conv2d + batch norm + (optionally) SiLU, which
    FlexFlow builds as three separate layers, so a ``Conv``'s output is
    FlexFlow's ``<path>.act`` when the activation is enabled and ``<path>.bn``
    when it is not.  Everything else (a bare ``nn.Conv2d`` in the detection
    head, ``nn.Upsample``, ``Concat``) is a single FlexFlow layer with the same
    name as the module.
    """
    result = {}
    for path, module in model.named_modules():
        if not path:
            continue
        if isinstance(module, Conv):
            suffix = ".bn" if isinstance(module.act, nn.Identity) else ".act"
            result[path] = path + suffix
        elif isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Upsample, Concat)):
            result[path] = path
    return result


def run_reference(model, inputs, capture=()):
    """Run the model's forward pass and return its outputs plus any captures.

    ``capture`` is a collection of *FlexFlow* tensor names whose values should
    be recorded.  Returns ``(outputs, captured)`` where ``outputs`` holds
    ``model.23.boxes`` and ``model.23.scores`` (the one2many head outputs, which
    are what FlexFlow's non-end2end YOLOv10 computes) and ``captured`` maps
    FlexFlow tensor names to CPU tensors.
    """
    wanted = set(capture)
    module_outputs = get_module_output_names(model)

    captured = {}
    handles = []

    def make_hook(ff_name):
        def hook(_module, _args, output):
            captured[ff_name] = output.detach().to("cpu", torch.float32)

        return hook

    for path, ff_name in module_outputs.items():
        if ff_name in wanted:
            handles.append(
                model.get_submodule(path).register_forward_hook(make_hook(ff_name))
            )

    try:
        preds = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    # v10Detect is an end-to-end head, so in training mode it returns both the
    # one2many and one2one predictions.  FlexFlow builds the non-end2end model,
    # which corresponds to the one2many branch.
    one2many = preds["one2many"] if "one2many" in preds else preds
    outputs = {
        "model.23.boxes": one2many["boxes"].detach().to("cpu", torch.float32),
        "model.23.scores": one2many["scores"].detach().to("cpu", torch.float32),
    }
    return outputs, captured


def _ff_output_name(path, module):
    """FlexFlow tensor name holding the output of the module at ``path``."""
    if isinstance(module, Conv):
        return path + (".bn" if isinstance(module.act, nn.Identity) else ".act")
    if isinstance(module, (C2f, SPPF, PSA, SCDown)):
        # These all end in a `cv2` convolution, which is what FlexFlow returns.
        return _ff_output_name(path + ".cv2", module.cv2)
    if isinstance(module, (nn.Upsample, Concat)):
        return path
    raise ValueError(f"don't know the FlexFlow output name for {path} ({type(module).__name__})")


def get_backbone_layer_outputs(model):
    """Map each backbone layer index to the FlexFlow name of its output tensor.

    The backbone is ``model.model[0]`` through ``model.model[22]``; index 23 is
    the detection head, whose outputs are ``model.23.boxes`` and
    ``model.23.scores``.
    """
    result = {}
    for i, module in enumerate(model.model):
        path = f"model.{i}"
        if i == len(model.model) - 1:
            continue  # the detection head
        result[i] = _ff_output_name(path, module)
    return result


def get_layer_inputs(model):
    """Map each layer index to the indices of the layers feeding it.

    ``-1`` means the previous layer; this is ultralytics' own ``from`` wiring,
    recorded on each module by ``parse_model``.
    """
    result = {}
    for i, module in enumerate(model.model):
        f = module.f
        result[i] = [f] if isinstance(f, int) else list(f)
    return result

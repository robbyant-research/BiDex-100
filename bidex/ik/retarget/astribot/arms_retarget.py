"""astribot ArmRetarget: thin subclass of the shared SLSQP arm-IK base
(``arm_retarget_base.py``, vendored next to this file). Resolves relative
config paths against THIS package dir so ``config_astribot.yaml`` is found
next to this file."""
import os
from bidex.ik.retarget.astribot.arm_retarget_base import ArmRetarget as _BaseArmRetarget

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ArmRetarget(_BaseArmRetarget):
    def __init__(self, config_file="config_astribot.yaml"):
        super().__init__(config_file=config_file)

    def load_config(self, config_file):
        if not os.path.isabs(config_file):
            config_file = os.path.join(_THIS_DIR, config_file)
        import yaml
        with open(config_file, "r") as fh:
            return yaml.safe_load(fh)
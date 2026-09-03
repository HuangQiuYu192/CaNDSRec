from .BERT4Rec import BERT4Rec
from .CANDSSASRec import CANDSSASRec
from .CANDSFMLPRec import CANDSFMLPRec
from .CANDSWEARec import CANDSWEARec
from .CalibratedCANDSSASRec import CalibratedCANDSSASRec
from .DataAwareTempCANDSSASRec import DataAwareTempCANDSSASRec
from .LearnableTempCANDSSASRec import LearnableTempCANDSSASRec
from .LinearItemCANDSSASRec import LinearItemCANDSSASRec
from .TailCLCalibratedCANDSSASRec import TailCLCalibratedCANDSSASRec
from .SASRec import SASRec
from .GRU4Rec import GRU4Rec
from .WEARec import WEARec
from .BSARec import BSARec
from .FMLPRec import FMLPRec
from .CL4SRec import CL4SRec
from .DuoRec import DuoRec
from .CoSeRec import CoSeRec
from .ICLRec import ICLRec
from .IOCRec import IOCRec
from .ICSRec import ICSRec
from .IOCRec import IOCRec
from .IDURL import IDURL

MODEL_DICT = {
    "GRU4Rec": GRU4Rec,
    "SASRec": SASRec,
    "CANDSSASRec": CANDSSASRec,
    "CANDSFMLPRec": CANDSFMLPRec,
    "CANDSWEARec": CANDSWEARec,
    "CalibratedCANDSSASRec": CalibratedCANDSSASRec,
    "DataAwareTempCANDSSASRec": DataAwareTempCANDSSASRec,
    "LearnableTempCANDSSASRec": LearnableTempCANDSSASRec,
    "LinearItemCANDSSASRec": LinearItemCANDSSASRec,
    "TailCLCalibratedCANDSSASRec": TailCLCalibratedCANDSSASRec,
    "BERT4Rec": BERT4Rec,
    "CL4SRec": CL4SRec,
    "CoSeRec": CoSeRec,
    "DuoRec": DuoRec,
    "ICLRec": ICLRec,
    "ICSRec": ICSRec,
    "IOCRec": IOCRec,
    "IDURL": IDURL,
    "FMLPRec": FMLPRec,
    "BSARec": BSARec,
    "WEARec": WEARec,
}


def get_model_class(model_name):
    model_class = MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_class

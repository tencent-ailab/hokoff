### 1v1 ###
from .OneBCLearner import OneBCLearner
from .OneCQLLearner import OneCQLLearner
from .OneTD3BCLearner import OneTD3BCLearner
from .OneIQLLearner import OneIQLLearner
from .OneQMIXLearner import OneQMIXLearner

REGISTRY = {}

### 1v1 ###
REGISTRY["1v1bc"] = OneBCLearner
REGISTRY["1v1cql"] = OneCQLLearner
REGISTRY["1v1td3bc"] = OneTD3BCLearner
REGISTRY["1v1iql"] = OneIQLLearner
REGISTRY["1v1qmix"] = OneQMIXLearner

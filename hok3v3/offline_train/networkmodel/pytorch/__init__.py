from .BCLearner import BCLearner
from .CQLLearner import CQLLearner
from .ICQLearner import ICQLearner
from .MAICQLearner import MAICQLearner

from .OMARLearner import OMARLearner
from .QMIXLearner import QMIXLearner


REGISTRY = {}

REGISTRY["bc"] = BCLearner
REGISTRY["cql"] = CQLLearner
REGISTRY["indbc"] = BCLearner
REGISTRY["indcql"] = CQLLearner
REGISTRY["icq"] = ICQLearner
REGISTRY["indicq"] = ICQLearner
REGISTRY["indmaicq"] = MAICQLearner
REGISTRY["maicq"] = MAICQLearner
REGISTRY["omar"] = OMARLearner
REGISTRY["indomar"] = OMARLearner
REGISTRY["qmix"] = QMIXLearner
REGISTRY["indqmix"] = QMIXLearner

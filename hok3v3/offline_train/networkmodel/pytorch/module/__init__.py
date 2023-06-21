from .BaseModel import BaseModel
from .IndModel import IndModel

INFER_REGISTRY = {}

INFER_REGISTRY["bc"] = BaseModel
INFER_REGISTRY["cql"] = BaseModel
INFER_REGISTRY["indbc"] = IndModel
INFER_REGISTRY["indcql"] = IndModel
INFER_REGISTRY["icq"] = BaseModel
INFER_REGISTRY["indicq"] = IndModel
INFER_REGISTRY["indmaicq"] = IndModel
INFER_REGISTRY["indomar"] = IndModel
INFER_REGISTRY["indqmix"] = IndModel
INFER_REGISTRY["maicq"] = BaseModel
INFER_REGISTRY["omar"] = BaseModel
INFER_REGISTRY["qmix"] = BaseModel

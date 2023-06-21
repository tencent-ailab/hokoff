from .OneBaseModel import OneBaseModel

INFER_REGISTRY = {}

INFER_REGISTRY["1v1bc"] = OneBaseModel
INFER_REGISTRY["1v1cql"] = OneBaseModel
INFER_REGISTRY["1v1td3bc"] = OneBaseModel
INFER_REGISTRY["1v1iql"] = OneBaseModel
INFER_REGISTRY["1v1qmix"] = OneBaseModel
INFER_REGISTRY["1v1sacbc"] = OneBaseModel

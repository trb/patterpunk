import copy
from typing import Iterable, TypeVar

ModelType = TypeVar("ModelType")


def deepcopy_sharing_clients(
    model: ModelType, memo_dict: dict, client_attribute_names: Iterable[str]
) -> ModelType:
    # Chat.add_message deep-copies the chat, model included, on every message.
    # A fixed constructor call would drop the subclass and every attribute it does
    # not name.
    # SDK clients cannot be deep-copied because their connection pools hold locks.
    # boto3 documents low-level clients as safe to share across threads.
    copied_model = copy.copy(model)
    memo_dict[id(model)] = copied_model
    shared_attribute_names = set(client_attribute_names)
    for attribute_name, attribute_value in vars(model).items():
        if attribute_name in shared_attribute_names:
            continue
        setattr(copied_model, attribute_name, copy.deepcopy(attribute_value, memo_dict))
    return copied_model

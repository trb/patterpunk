import logging

logger = logging.getLogger("patterpunk")
logger_llm = logging.getLogger("patterpunk.llm")

logger.setLevel(logging.DEBUG)
logger_llm.setLevel(logging.DEBUG)

logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)

logger_llm_handler = logging.StreamHandler()
logger_llm.addHandler(logger_llm_handler)

logger_llm.propagate = False

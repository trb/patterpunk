from dataclasses import dataclass
from typing import List, Union
from patterpunk.llm.agent import Agent
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Parallel:
    agents: List[Agent]


class AgentChain:
    def __init__(self, steps: List[Union[Agent, Parallel]]):
        self.steps = steps

    def execute(self, initial_input):
        current_input = initial_input
        for step in self.steps:
            if isinstance(step, Agent):
                current_input = step.execute(current_input)
            elif isinstance(step, Parallel):
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(agent.execute, current_input)
                        for agent in step.agents
                    ]
                    current_input = [future.result() for future in futures]
        return current_input

"""
synthesis_evaluation_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
from typing import Optional
import json

# Third party imports
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM

# Custom imports
from ......utils.configs.schema_config import ExtractionSchemaConfig, load_schema_config
from ......utils.schema_tools import (
    build_evaluation_model,
    evaluation_prompt_values,
    fill_schema_placeholders,
)


@CrewBase
class SynthesisEvaluationCrew:
    """
    A CrewAI crew for evaluating synthesis data using AI agent reasoning.
    This crew uses binary matching (yes/no) rather than semantic similarity or exact matching.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        schema_config: Optional[ExtractionSchemaConfig] = None,
    ):
        """
        Args:
            llm (Optional[LLM], optional): LLM instance for the agent. Defaults to LLM(model="o3-mini").
            schema_config (Optional[ExtractionSchemaConfig], optional): Extraction schema defining the evaluated
                fields. Defaults to the bundled default schema.
        """
        self.llm = llm or LLM(model="o3-mini")
        self.schema_config = schema_config or load_schema_config()
        fields = self.schema_config.fields("synthesis", final_only=True)
        self.output_model = build_evaluation_model(fields, "synthesis")
        self.schema_prompt_values = evaluation_prompt_values(fields, "synthesis")

    @agent
    def synthesis_evaluator_agent(self) -> Agent:
        """Agent that evaluates synthesis data with binary decisions.

        Returns:
            Agent: Configured CrewAI agent using the synthesis evaluator prompt config.
        """
        return Agent(
            config=fill_schema_placeholders(
                self.agents_config["synthesis_evaluator_agent"],
                self.schema_prompt_values,
            ),
            llm=self.llm,
        )

    @task
    def evaluate_synthesis_data_task(self) -> Task:
        """Task for evaluating synthesis data with binary decisions.

        Returns:
            Task: Configured CrewAI task that outputs the synthesis evaluation pydantic model.
        """
        return Task(
            config=fill_schema_placeholders(
                self.tasks_config["evaluate_synthesis_data_task"],
                self.schema_prompt_values,
            ),
            output_pydantic=self.output_model,
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the synthesis evaluation crew.

        Returns:
            Crew: Sequential-process crew wiring the synthesis evaluator agent to its task.
        """
        return Crew(
            agents=[self.synthesis_evaluator_agent()],
            tasks=[self.evaluate_synthesis_data_task()],
            verbose=True,
            process=Process.sequential,
        )

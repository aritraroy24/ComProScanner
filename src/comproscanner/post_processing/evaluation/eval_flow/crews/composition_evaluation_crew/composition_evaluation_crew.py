"""
composition_evaluation_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-04-2025
"""

# Standard library imports
from typing import Dict, Optional, List, Tuple, Type

# Third party imports
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Custom imports
from ......utils.configs.schema_config import ExtractionSchemaConfig, load_schema_config
from ......utils.schema_tools import (
    build_evaluation_model,
    evaluation_prompt_values,
    fill_schema_placeholders,
)


class ThresholdToolInput(BaseModel):
    """Input schema for GetValueErrorThresholdTool."""

    reference_value: str = Field(
        ...,
        description="The ground-truth numeric property value as a string, e.g. '150' or '-300.5'.",
    )


class GetValueErrorThresholdTool(BaseTool):
    """
    Returns the allowed absolute error tolerance for a numeric ground-truth property value.

    When an evaluator has been configured with a ``value_error_thresholds`` mapping, this
    tool lets the agent look up how much the extracted value is allowed to differ from the
    ground-truth value before the comparison is counted as a mismatch.

    The returned string is one of:
    - ``"threshold:<N>"`` — the extracted value matches if |extracted - reference| <= N.
    - ``"exact"`` — no tolerance configured; use exact matching (within 1e-6).
    """

    name: str = "get_value_error_threshold"
    description: str = (
        "Look up the allowed absolute error tolerance for a numeric ground-truth property value. "
        "Call this tool with the ground-truth (reference) value before deciding whether a "
        "numeric property value extracted from the test data is a match. "
        "If the tool returns 'threshold:<N>', the test value is accepted if "
        "|test_value - reference_value| <= N. If the tool returns 'exact', require exact equality."
    )
    args_schema: Type[BaseModel] = ThresholdToolInput

    # Each element is (lo, hi, threshold) with lo <= hi
    thresholds_list: List[Tuple[float, float, float]] = Field(default_factory=list)

    def _run(self, reference_value: str) -> str:
        try:
            ref_num = float(reference_value)
        except (ValueError, TypeError):
            return "exact: reference value is not numeric"

        for lo, hi, threshold in self.thresholds_list:
            if lo <= ref_num <= hi:
                return f"threshold:{threshold}"

        return "exact"


@CrewBase
class CompositionEvaluationCrew:
    """
    A CrewAI crew for evaluating composition data using AI agent reasoning.
    This crew uses binary matching (yes/no) rather than semantic similarity or exact matching.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        value_error_thresholds: Optional[Dict] = None,
        schema_config: Optional[ExtractionSchemaConfig] = None,
    ):
        """
        Args:
            llm: LLM instance for the agent.
            value_error_thresholds: Mapping of ``(min, max)`` tuples to absolute error
                tolerances for numeric property-value comparisons.  When provided, the
                ``get_value_error_threshold`` tool is added to the evaluator agent so that
                it can look up the tolerance before deciding on value matches.  Example::

                    {
                        (-50, 50): 0.5,
                        (-150, 150): 1,
                        (-1000, 1000): 2,
                        (float('-inf'), float('inf')): 5,
                    }
            schema_config: Extraction schema defining the evaluated fields. Defaults to the bundled default schema.
        """
        self.llm = llm or LLM(model="o3-mini")
        self.schema_config = schema_config or load_schema_config()
        fields = self.schema_config.fields("composition", final_only=True)
        self.output_model = build_evaluation_model(fields, "composition")
        self.schema_prompt_values = evaluation_prompt_values(fields, "composition")
        # Convert the dict to an internal list of (lo, hi, threshold) triples sorted by
        # span ascending so the narrowest matching range always wins (layered semantics).
        self._thresholds_list: List[Tuple[float, float, float]] = []
        if value_error_thresholds:
            triples = []
            for range_key, threshold in value_error_thresholds.items():
                lo = min(range_key)
                hi = max(range_key)
                triples.append((lo, hi, float(threshold)))
            self._thresholds_list = sorted(triples, key=lambda t: t[1] - t[0])

    @agent
    def composition_evaluator_agent(self) -> Agent:
        """Agent that evaluates composition data with binary decisions."""
        tools = []
        if self._thresholds_list:
            tools = [
                GetValueErrorThresholdTool(thresholds_list=self._thresholds_list)
            ]
        return Agent(
            config=fill_schema_placeholders(
                self.agents_config["composition_evaluator_agent"],
                self.schema_prompt_values,
            ),
            llm=self.llm,
            tools=tools,
        )

    @task
    def evaluate_composition_data_task(self) -> Task:
        """Task for evaluating composition data with binary decisions."""
        return Task(
            config=fill_schema_placeholders(
                self.tasks_config["evaluate_composition_data_task"],
                self.schema_prompt_values,
            ),
            output_pydantic=self.output_model,
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the composition evaluation crew."""
        return Crew(
            agents=[self.composition_evaluator_agent()],
            tasks=[self.evaluate_composition_data_task()],
            verbose=True,
            process=Process.sequential,
        )

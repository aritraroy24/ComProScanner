"""
synthesis_extraction_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import os
from typing import Optional
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .....utils.configs.llm_config import LLMConfig
from .....utils.configs.schema_config import ExtractionSchemaConfig, load_schema_config
from .....utils.schema_tools import (
    build_output_model,
    fill_schema_placeholders,
    schema_prompt_values,
)


@CrewBase
class SynthesisExtractionCrew:
    """Synthesis Extraction Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        doi: str = None,
        llm: Optional[LLM] = None,
        output_log_folder: Optional[str] = None,
        task_output_folder: Optional[str] = None,
        is_log_json: bool = False,
        verbose: Optional[bool] = True,
        schema_config: Optional[ExtractionSchemaConfig] = None,
    ):
        """Initialize the synthesis extraction crew for a given article.

        Args:
            doi (str, required): DOI of the article to be processed.
            llm (LLM, optional): LLM instance to use. Defaults to None, which builds one from LLMConfig.
            output_log_folder (str, optional): Base folder for saving crew logs under {folder}/{doi}. Logs are JSON if is_log_json is True, otherwise .txt. Defaults to None.
            task_output_folder (str, optional): Base folder for saving task outputs under {folder}/{doi} as .txt files. Defaults to None.
            is_log_json (bool, optional): Whether to save logs in JSON format. Defaults to False.
            verbose (bool, optional): Verbosity flag passed to the agent and crew. Defaults to True.
            schema_config (ExtractionSchemaConfig, optional): Extraction schema. Defaults to the bundled default schema.

        Raises:
            ValueError: If doi is not provided.
        """
        if doi is None:
            raise ValueError("DOI must be provided")
        self.doi = doi
        # Use provided LLM or create default one
        self.llm = llm if llm is not None else LLMConfig().get_llm()
        self.output_log_folder = output_log_folder
        self.task_output_folder = task_output_folder
        self.is_log_json = is_log_json
        self.verbose = verbose
        self.schema_config = schema_config or load_schema_config()
        fields = self.schema_config.fields("synthesis")
        self.output_model = build_output_model(
            fields,
            wrapper_key="synthesis_extracted_data",
            model_name="ExtractedSynthesisData",
            inner_model_name="DetailedSynthesisData",
            model_doc="Extracted Synthesis Data",
            inner_model_doc="Detailed Synthesis Data",
        )
        self.schema_prompt_values = schema_prompt_values(fields, stage="extraction")

        # Initialize output file paths as None
        self.output_log_file = None
        self.task_output_file = None

        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_output_log_folder):
                os.makedirs(final_output_log_folder)
            if self.is_log_json:
                self.output_log_file = (
                    f"{final_output_log_folder}/synthesis_extractor_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/synthesis_extractor_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = (
                f"{final_task_output_folder}/synthesis_extractor_task_output.txt"
            )

    # Agents
    @agent
    def synthesis_extractor(self) -> Agent:
        """Build the synthesis extraction agent.

        Returns:
            Agent: configured agent for extracting synthesis data.
        """
        return Agent(
            config=fill_schema_placeholders(
                self.agents_config["synthesis_extractor"], self.schema_prompt_values
            ),
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def extract_synthesis_data(self) -> Task:
        """Build the synthesis data extraction task, writing to a file if configured.

        Returns:
            Task: configured task producing output_model, with output_file set when task_output_file is set.
        """
        if self.task_output_file:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_synthesis_data"], self.schema_prompt_values
                ),
                output_pydantic=self.output_model,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_synthesis_data"], self.schema_prompt_values
                ),
                output_pydantic=self.output_model,
            )

    # Crew
    @crew
    def crew(self) -> Crew:
        """Build the sequential crew, logging to a file if configured.

        Returns:
            Crew: configured crew running the agents and tasks sequentially.
        """
        if self.output_log_file:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
                output_log_file=self.output_log_file,
            )
        else:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
            )

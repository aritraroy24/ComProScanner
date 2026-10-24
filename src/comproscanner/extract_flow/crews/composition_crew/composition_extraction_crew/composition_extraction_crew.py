"""
composition_extraction_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-03-2025
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
from ....tools.graph_extractor_tool import GraphExtractorTool
from ....tools.equation_tool import EquationTool


@CrewBase
class CompositionExtractionCrew:
    """Composition Crew"""

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
        vlm_model: str = "gemini/gemini-3-flash-preview",
        related_figures_base_path: str = "results/related_figures",
        main_extraction_keyword: str = "",
        formula_instruction: str = None,
        equation_model: str = None,
        schema_config: Optional[ExtractionSchemaConfig] = None,
    ):
        """
        Initialize the MaterialsDataIdentifierCrew.

        Parameters:
        - doi (str): DOI of the article to be processed. This is a required parameter.
        - llm: Optional LLM instance. If not provided, will use default configuration
        - output_log_folder (str, optional): Set to True to save logs inside {provided folder}/{doi} folder as .txt files. Logs will be in JSON format if the is_log_json is True, otherwise .txt. Defaults to None.
        - task_output_folder (str, optional): Folder path for storing the task outputs as .txt files inside {provided foler}/{doi} folder. Defaults to None.
        - is_log_json (bool, optional): Flag to save logs in JSON format. Defaults to False.
        - verbose: Optional boolean for verbosity. Default is True.
        - vlm_model (str, optional): Vision LLM model for graph extraction. Defaults to "gemini/gemini-3-flash-preview".
        - related_figures_base_path (str, optional): Base path for saved figures. Defaults to "results/related_figures".
        - main_extraction_keyword (str, optional): Property keyword used to label axes in the VLM extraction prompt. Defaults to "".
        - equation_model (str, optional): Explicit litellm model name for EquationTool. Defaults to None (auto-select).
        - schema_config (ExtractionSchemaConfig, optional): Extraction schema. Defaults to the bundled default schema.
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
        self.vlm_model = vlm_model
        self.related_figures_base_path = related_figures_base_path
        self.main_extraction_keyword = main_extraction_keyword
        self.formula_instruction = formula_instruction
        self.equation_model = equation_model
        self.schema_config = schema_config or load_schema_config()
        fields = self.schema_config.fields("composition")
        self.output_model = build_output_model(
            fields,
            wrapper_key="composition_extracted_data",
            model_name="ExtractedCompositionPropertyData",
            inner_model_name="DetailedCompositionPropertyData",
            model_doc="Extracted Composition Property",
            inner_model_doc="Detailed Composition Property",
        )
        self.schema_prompt_values = schema_prompt_values(fields, stage="extraction")

        # Initialize output file paths as None
        self.output_log_file = None
        self.task_output_file = None

        final_task_output_folder = f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_output_log_folder):
                os.makedirs(final_output_log_folder)
            if self.is_log_json:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_extractor_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_extractor_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = f"{final_task_output_folder}/composition_property_extractor_task_output.txt"

    # Agents
    @agent
    def composition_property_extractor(self) -> Agent:
        equation_tool_kwargs = {
            "related_figures_base_path": self.related_figures_base_path,
        }
        if self.formula_instruction is not None:
            equation_tool_kwargs["formula_instruction"] = self.formula_instruction
        if self.equation_model is not None:
            equation_tool_kwargs["equation_model"] = self.equation_model
        equation_tool = EquationTool(**equation_tool_kwargs)
        graph_tool = GraphExtractorTool(
            vlm_model=self.vlm_model,
            related_figures_base_path=self.related_figures_base_path,
            vlm_property_name=self.main_extraction_keyword,
        )
        return Agent(
            config=fill_schema_placeholders(
                self.agents_config["composition_property_extractor"], self.schema_prompt_values
            ),
            tools=[equation_tool, graph_tool],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def extract_composition_property(self) -> Task:
        if self.task_output_file:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_composition_property"], self.schema_prompt_values
                ),
                output_pydantic=self.output_model,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_composition_property"], self.schema_prompt_values
                ),
                output_pydantic=self.output_model,
            )

    # Crew
    @crew
    def crew(self) -> Crew:
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

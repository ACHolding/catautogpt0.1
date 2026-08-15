from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.prompts.generator import PromptGenerator

DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use based on the given goals "
    "and the progress you have made so far. "
    "Prefer a concrete action that advances the goals "
    "(for example write_to_file, read_file, execute_python_file, or browse_website). "
    "Do not use goals_accomplished or task_complete until the goals are actually done. "
    "Respond using the JSON schema specified previously:"
)


def build_default_prompt_generator(config: Config) -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and best practices.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Initialize the PromptConfig object and load the file set in the main config (default: prompts_settings.yaml)
    prompt_config = PromptConfig(config.prompt_settings_file)

    # Add constraints to the PromptGenerator object
    for constraint in prompt_config.constraints:
        prompt_generator.add_constraint(constraint)

    # Add resources to the PromptGenerator object
    for resource in prompt_config.resources:
        prompt_generator.add_resource(resource)

    # Add best practices to the PromptGenerator object
    for best_practice in prompt_config.best_practices:
        prompt_generator.add_best_practice(best_practice)

    return prompt_generator

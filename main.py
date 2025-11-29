import os
import time
from dotenv import load_dotenv
import json
import google.generativeai as genai
import PIL.Image


# --- Load prompts from JSON file ---
with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
LLM1_PROMPT = prompts["LLM1_PROMPT"]
LLM2_PROMPT = prompts["LLM2_PROMPT"]
ESTIMATION_PROMPT = prompts["ESTIMATION_PROMPT"]
SIMULATION_PROMPT = prompts["SIMULATION_PROMPT"]

# --- Load ODE equations from JSON file ---
with open("ode.json", "r", encoding="utf-8") as f:
    ode_equations = json.load(f)
hiv_ode = ode_equations["hivModel"]
covid_ode = ode_equations["covidModel"]
simple_ode = ode_equations["simpleModel"]

# --- Load models from JSON file ---
with open("models.json", "r", encoding="utf-8") as f:
    models = json.load(f)
hivModel = models["hivModel"]
covidModel = models["covidModel"]
simpleModel = models["simpleModel"]


try:
    # Load the API key from environment variables
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("FATAL ERROR: 'GOOGLE_API_KEY' environment variable not set.")
    print("Please set it before running the script.")
    exit()
model = genai.GenerativeModel('gemini-2.5-pro')









def generate_seirmodel_from_image(image_path: str, user_input: str, output_fileName: str) -> str:
    """
    Generates a SEIR model in XML format using a two-stage LLM process with multimodal input,
    and saves the full interaction log.

    Args:
        image_path (str): Path to the epidemiological model diagram image.
        user_input (str): Text containing tabular data (compartments, flows, variables).
        output_file_name (str): The desired name for the output XML file (e.g., "hiv_model.xml").

    Returns:
        str: A success message with the output file path, or an error message.
    """
    try:
        # Load the language specifications from a JSON file and convert to a pretty string
        with open(METAMODEL_FILENAME, "r", encoding="utf-8") as f:
            lang_specs_json = json.load(f)
        lang_specs = json.dumps(lang_specs_json, indent=2)
        
        # Load the image using the modern PIL library
        img_path = os.path.join(os.path.dirname(__file__), "diagrams", image_path)
        img = PIL.Image.open(img_path)

    except FileNotFoundError as err:
        print(f"Error: A required file was not found - {err}")
        return f"Error: A required file was not found - {err}"
    except Exception as err:
        print(f"An unexpected error occurred while reading files: {err}")
        return f"An unexpected error occurred while reading files: {err}"


    print(f"'{image_path}' loaded successfully.")
    print(f"'{METAMODEL_FILENAME}' loaded successfully.")
    # --- Construct the final prompt text ---
    separator = "\n" + "*" * 80 + "\n"


    llm1_input = (
        f"{separator}"
        f"PROMPT: \n{LLM1_PROMPT.strip()}\n"
        f"{separator}"
        f"METAMODEL: \n{lang_specs.strip()}\n"
        f"{separator}"
        f"USER_INPUT: \n{user_input.strip()}"
    )

    llm1 = model.generate_content([
        img,                 # The image object
        llm1_input.strip()  # The text part of the prompt
    ]).text.strip()


    llm2_input = (
        f"{separator}"
        f"PROMPT:\n{LLM2_PROMPT.strip()}\n"
        f"{separator}"
        f"USER INPUT:\n{user_input.strip()}\n"
        f"{separator}"
        f"STRUCTURALLY CORRECT SEIRMODEL FILE:\n{llm1.strip()}\n"
        f"{separator}"
    )

    # --- Call the modern API with a list of parts (image and text) ---
    llm2 = model.generate_content(llm2_input.strip()).text.strip()
    
    # --- Format and save the output ---
    output_content =(
        f"LLM1 PROMPT:\n{LLM1_PROMPT.strip()}"
        f"{separator}"
        f"METAMODEL:\n{lang_specs.strip()}"
        f"{separator}"
        f"User Input:\n{user_input.strip()}"
        f"{separator}"
        f"Image is also provided with the above textual input!"
        f"{separator}"
        f"LLM1 RESPONSE:\n {llm1}"
        f"{separator}"
        f"{separator}"
        f"LLM2 PROMPT:\n{LLM2_PROMPT.strip()}"
        f"{separator}"
        f"USER INPUT: \n{user_input}"
        f"{separator}"
        f"LLM1'S RESPONSE: \n{llm1}"
        f"{separator}"
        f"LLM2'S RESPONSE:\n{llm2}"
    )

    try:
        # Ensure the output directory exists
        output_dir = os.path.join(os.path.dirname(output_fileName), "prompt_sample")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(output_fileName))
        with open(output_file, "w", encoding="utf-8") as tf:
            tf.write(output_content)
    except IOError as e:
        return f"Error writing to output file '{output_fileName}': {e}"

    print(f"SEIR model successfully written to {output_fileName}")



def estimate_and_simulate(ode_equations: str, estimation_image_path: str, output_fileName: str) -> str:
    """
    Estimates missing parameters and generates a Python simulation script for an ODE model.

    Args:
        ode_equations (str): The ODE equations for the model.
        estimation_image_path (str): Path to the result graph image for estimation.
        output_fileName (str): The desired name for the output Python file (e.g., "simulation.py").

    Returns:
        str: A success message with the output file path, or an error message.
    """
    try:
        # Load the result graph image
        img_path = os.path.join(os.path.dirname(__file__), "diagrams", estimation_image_path)
        img = PIL.Image.open(img_path)

    except FileNotFoundError as err:
        print(f"Error: A required file was not found - {err}")
        return f"Error: A required file was not found - {err}"
    except Exception as err:
        print(f"An unexpected error occurred while reading files: {err}")
        return f"An unexpected error occurred while reading files: {err}"

    print(f"'{estimation_image_path}' loaded successfully.")
    
    # --- Stage 1: Estimation ---
    separator = "\n" + "*" * 80 + "\n"

    estimation_input = (
        f"{separator}"
        f"PROMPT: \n{ESTIMATION_PROMPT.strip()}\n"
        f"{separator}"
        f"ODE_EQUATIONS: \n{ode_equations.strip()}\n"
        f"{separator}"
    )

    estimated_values = model.generate_content([
        img,                 # The image object
        estimation_input.strip()  # The text part of the prompt
    ]).text.strip()


    # --- Stage 2: Simulation ---
    simulation_input = (
        f"{separator}"
        f"PROMPT:\n{SIMULATION_PROMPT.strip()}\n"
        f"{separator}"
        f"ODE_EQUATIONS:\n{ode_equations.strip()}\n"
        f"{separator}"
        f"ESTIMATED_VALUES:\n{estimated_values.strip()}\n"
        f"{separator}"
    )

    simulation_script = model.generate_content(simulation_input.strip()).text.strip()
    
    # --- Format and save the output ---
    try:
        # Ensure the output directory exists
        output_dir = "simulation_scripts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(output_fileName))
        with open(output_file, "w", encoding="utf-8") as tf:
            tf.write(simulation_script)
    except IOError as e:
        return f"Error writing to output file '{output_fileName}': {e}"

    print(f"Simulation script successfully written to {output_fileName}")





generate_seirmodel_from_image("hivModel(paper).jpeg", hivModel, "finalHivModel.txt")

time.sleep(10) #Helps not to not burn tokens too quickly

generate_seirmodel_from_image("covidModel(paper).png", covidModel, "finalCovidModel.txt")

time.sleep(10) #Helps not to not burn tokens too quickly

generate_seirmodel_from_image("simpleSIRModel(paper).png", simpleModel, "finalSimpleModel.txt")

time.sleep(10) #Helps not to not burn tokens too quickly

estimate_and_simulate(hiv_ode, "hivModel(paper).jpeg", "hiv_simulation.py")

time.sleep(10) #Helps not to not burn tokens too quickly

estimate_and_simulate(covid_ode, "covidModel(paper).png", "covid_simulation.py")

time.sleep(10) #Helps not to not burn tokens too quickly

estimate_and_simulate(simple_ode, "simpleSIRModel(paper).png", "simple_simulation.py")

import os
import time
from dotenv import load_dotenv
import json
import google.generativeai as genai
import PIL.Image


# --- Load configuration files ---
def load_json_file(filename: str) -> dict:
    """Load and return JSON file contents."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{filename}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"ERROR: '{filename}' is not valid JSON.")
        raise


# Load prompts and models
prompts = load_json_file("prompts.json")
LLM1_PROMPT = prompts["LLM1_PROMPT"]
LLM2_PROMPT = prompts["LLM2_PROMPT"]
SIMULATION_PROMPT = prompts["SIMULATION_PROMPT"]
METAMODEL_FILENAME = "metamodel.json"

models = load_json_file("models.json")
hivModel = models["hivModel"]
covidModel = models["covidModel"]
simpleModel = models["simpleModel"]
malariaModel = models["malariaModel"]


# --- Initialize Gemini API ---
try:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("FATAL ERROR: 'GEMINI_API_KEY' environment variable not set.")
    print("Please set it before running the script.")
    exit(1)

# Use the stable API that works
model = genai.GenerativeModel('gemini-2.5-pro')


def generate_seirmodel_from_image(
    image_path: str, 
    user_input: str, 
    output_fileName: str
) -> str:
    """
    Generates a SEIR model in XML format using a two-stage LLM process with multimodal input,
    and saves the full interaction log.

    Args:
        image_path: Path to the epidemiological model diagram image (relative to 'diagrams' folder).
        user_input: Text containing tabular data (compartments, flows, variables).
        output_fileName: The desired name for the output file (e.g., "hiv_model.txt").

    Returns:
        str: A success message with the output file path, or an error message.
    """
    try:
        # Load metamodel specifications
        with open(METAMODEL_FILENAME, "r", encoding="utf-8") as f:
            lang_specs_json = json.load(f)
        lang_specs = json.dumps(lang_specs_json, indent=2)
        
        # Load the image using PIL
        img_path = os.path.join(os.path.dirname(__file__), "diagrams", image_path)
        img = PIL.Image.open(img_path)

        print(f"'{image_path}' loaded successfully.")
        print(f"'{METAMODEL_FILENAME}' loaded successfully.")

    except FileNotFoundError as err:
        error_msg = f"Error: File not found - {err}"
        print(error_msg)
        return error_msg
    except Exception as err:
        error_msg = f"Unexpected error reading files: {err}"
        print(error_msg)
        return error_msg

    # --- Stage 1: Structural generation ---
    separator = "\n" + "*" * 80 + "\n"
    
    llm1_input = (
        f"{separator}"
        f"PROMPT: \n{LLM1_PROMPT.strip()}\n"
        f"{separator}"
        f"METAMODEL: \n{lang_specs.strip()}\n"
        f"{separator}"
        f"USER_INPUT: \n{user_input.strip()}"
    )

    # Generate content with image + text (old SDK style)
    llm1 = model.generate_content([
        img,                   # The PIL Image object
        llm1_input.strip()    # The text part of the prompt
    ]).text.strip()
    
    print("LLM1 response generated successfully.")

    # --- Stage 2: Refinement ---
    llm2_input = (
        f"{separator}"
        f"PROMPT:\n{LLM2_PROMPT.strip()}\n"
        f"{separator}"
        f"USER INPUT:\n{user_input.strip()}\n"
        f"{separator}"
        f"STRUCTURALLY CORRECT SEIRMODEL FILE:\n{llm1.strip()}\n"
        f"{separator}"
    )

    llm2 = model.generate_content(llm2_input.strip()).text.strip()
    print("LLM2 response generated successfully.")
    
    # --- Format and save the output ---
    output_content = (
        f"LLM1 PROMPT:\n{LLM1_PROMPT.strip()}"
        f"{separator}"
        f"METAMODEL:\n{lang_specs.strip()}"
        f"{separator}"
        f"User Input:\n{user_input.strip()}"
        f"{separator}"
        f"Image is also provided with the above textual input!"
        f"{separator}"
        f"LLM1 RESPONSE:\n{llm1}"
        f"{separator}"
        f"{separator}"
        f"LLM2 PROMPT:\n{LLM2_PROMPT.strip()}"
        f"{separator}"
        f"USER INPUT:\n{user_input}"
        f"{separator}"
        f"LLM1'S RESPONSE:\n{llm1}"
        f"{separator}"
        f"LLM2'S RESPONSE:\n{llm2}"
    )

    try:
        # Ensure the output directory exists
        output_dir = "prompt_sample"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_fileName)
        
        with open(output_file, "w", encoding="utf-8") as tf:
            tf.write(output_content)
        
        success_msg = f"SEIR model successfully written to {output_file}"
        print(success_msg)
        return success_msg
        
    except IOError as e:
        error_msg = f"Error writing to output file '{output_file}': {e}"
        print(error_msg)
        return error_msg


def simulate(ode_equations: str, output_fileName: str) -> str:
    """
    Generates a Python simulation script for an ODE model.

    Args:
        ode_equations: Identifier for the ODE equations (e.g., 'hiv_ode', 'covid_ode').
        output_fileName: The desired name for the output Python file (e.g., "simulation.py").

    Returns:
        str: A success message with the output file path, or an error message.
    """
    try:
        # Load ODE equations from JSON file
        ode_equations_data = load_json_file("ode.json")
        
        # Map identifier to actual equations
        ode_mapping = {
            "hiv_ode": ode_equations_data["hivModel"],
            "covid_ode": ode_equations_data["covidModel"],
            "simple_ode": ode_equations_data["simpleModel"]
        }
        
        ode_eq = ode_mapping.get(ode_equations)
        if ode_eq is None:
            error_msg = f"Unknown ODE equations identifier: {ode_equations}"
            print(error_msg)
            return error_msg
        
    except Exception as e:
        error_msg = f"Error loading ODE equations: {e}"
        print(error_msg)
        return error_msg
    
    # --- Generate simulation script ---
    separator = "\n" + "*" * 80 + "\n"
    
    simulation_input = (
        f"{separator}"
        f"PROMPT:\n{SIMULATION_PROMPT.strip()}\n"
        f"{separator}"
        f"ODE_EQUATIONS:\n{ode_eq.strip()}\n"
        f"{separator}"
    )

    simulation_script_response = model.generate_content(simulation_input.strip())
    simulation_script = simulation_script_response.text.strip()
    print("Simulation script generated successfully.")
    
    # --- Save the output ---
    try:
        output_dir = "simulation_scripts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_fileName)
        
        with open(output_file, "w", encoding="utf-8") as tf:
            tf.write(simulation_script)
        
        success_msg = f"Simulation script successfully written to {output_file}"
        print(success_msg)
        return success_msg
        
    except IOError as e:
        error_msg = f"Error writing to output file '{output_file}': {e}"
        print(error_msg)
        return error_msg


def main():
    """Main execution function with rate limiting."""
    
    # Generate SEIR models
    print("\n" + "="*80)
    print("Generating HIV Model...")
    print("="*80)
    generate_seirmodel_from_image("hivModel(paper).jpeg", hivModel, "finalHivModel.txt")
    
    time.sleep(10)  # Rate limiting

    print("\n" + "="*80)
    print("Generating COVID Model...")
    print("="*80)
    generate_seirmodel_from_image("covidModel(paper).png", covidModel, "finalCovidModel.txt")

    time.sleep(10)  # Rate limiting

    print("\n" + "="*80)
    print("Generating Simple SIR Model...")
    print("="*80)
    generate_seirmodel_from_image("simpleSIRModel(paper).png", simpleModel, "finalSimpleModel.txt")

    time.sleep(10)  # Rate limiting

    print("\n" + "="*80)
    print("Generating Malaria Model...")
    print("="*80)
    generate_seirmodel_from_image("malariaModel(paper).png", malariaModel, "finalSimpleModel.txt")
    
    # Uncomment to run simulations
    # time.sleep(10)
    # print("\n" + "="*80)
    # print("Generating HIV Simulation...")
    # print("="*80)
    # simulate("hiv_ode", "hiv_simulation.py")
    
    # time.sleep(10)
    # simulate("covid_ode", "covid_simulation.py")
    
    # time.sleep(10)
    # simulate("simple_ode", "simple_simulation.py")
    
    print("\n" + "="*80)
    print("All tasks completed!")
    print("="*80)


if __name__ == "__main__":
    main()

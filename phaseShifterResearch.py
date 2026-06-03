import os
from google import genai
from google.genai import types

client = genai.Client()

# System instruction focused on open-source silicon PDK design
SYSTEM_INSTRUCTION = """
You are a Principal Millimeter-Wave IC Design Engineer specializing in 5G FR2 Beamforming architectures and open-source EDA ecosystems.
Your goal is to conduct deep engineering research and produce an actionable design report for a 4-Bit Switched-Filter Phase Shifter.

This circuit must be designed specifically to leverage the open-source IHP-Open-PDK (SG13G2 130nm BiCMOS node). You will focus heavily on utilizing the 3.3V thick-oxide NMOS transistors ('nmos_thick_oxide') as the RF switching element. 

Your report must be mathematically rigorous and detail exact component setups so a secondary agent can write a functional Ngspice netlist using IHP primitive models.
"""

# Added 'r' before the quotes to make this a raw string and fix the \D syntax warning
USER_PROMPT = r"""
Please generate a comprehensive Millimeter-Wave IC Engineering Design Report based on the following structural specification JSON:

{
  "project": "28GHz_4Bit_Phase_Shifter",
  "topology_selection": "switched_filter_cascade",
  "frequency_ghz": 28.0,
  "bits": 4,
  "step_size_deg": 22.5,
  "switch_type": "nmos_thick_oxide",
  "target_metrics": {
    "insertion_loss_db": "< 6.0",
    "return_loss_db": "< -10.0",
    "rms_phase_error_deg": "< 3.0"
  }
}

Your report MUST include the following specific sections:

1. SWITCHED-FILTER TOPOLOGY BREAKDOWN
   - Explain how a single-bit stage alternates between a Low-Pass Network (which creates a phase delay) and a High-Pass Network (which creates a phase advance) using NMOS switches.
   - Detail the structural differences between T-networks and Pi-networks for this frequency, and recommend the best fit for silicon area optimization.
   - Provide a text block-diagram cascade layout of the 4 bits (180, 90, 45, 22.5 degrees).

2. MATHEMATICAL DETERMINATION OF L AND C
   - Provide the explicit equations used to calculate the required inductance (L) and capacitance (C) values for a given phase shift (\Delta\phi) in a 50-Ohm environment at 28 GHz.
   - State the ideal component values calculated for the 180-degree and 22.5-degree bit networks.

3. IHP SG13G2 NMOS MODELING
   - Detail how the 'nmos_thick_oxide' transistor behaves in SPICE as an RF switch. 
   - Explain the trade-off of using thick-oxide (3.3V) versus thin-oxide (1.2V) regarding Ron (on-resistance) and Coff (off-capacitance) at 28 GHz.
   - Describe the typical subcircuit or primitive model structure used in Ngspice for IHP MOS devices.

4. SIMULATION AND OPTIMIZATION SETUP
   - Outline how the next agent should structure the Ngspice simulation file to calculate S-parameters (S11 and S21) across all 16 digital switching states to check against our targets.
"""

print("📡 Launching 28 GHz Switched-Filter Research Agent (via Gemini Flash)...")

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.2,
)

try:
    # Switched model from gemini-2.5-pro to gemini-2.5-flash to avoid quota limits
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=USER_PROMPT,
        config=config
    )
    
    output_filename = "mmwave_phase_shifter_report.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    print(f"\n✅ Research Complete! Deep silicon report saved to: {output_filename}")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
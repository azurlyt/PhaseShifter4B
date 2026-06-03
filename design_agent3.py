import os
import subprocess
import json
import numpy as np
from scipy.optimize import minimize
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError

# =====================================================================
# 1. Environment & Path Setup
# =====================================================================
PDK_ROOT = r"C:\IHP-Open-PDK"
NGSPICE_EXE = r"C:\Spice64\bin\ngspice.exe"
MODEL_INCLUDE_PATH = os.path.join(PDK_ROOT, "ihp-sg13g2", "libs.tech", "ngspice", "models", "cornerMOShv.lib")

client = genai.Client()

class SupervisorDecision(BaseModel):
    topology: str             
    initial_c_val_ff: float   
    initial_l_val_ph: float   
    initial_w_switch_u: float 
    reasoning: str            

# =====================================================================
# 2. Dynamic Netlist Compiler (True Branchline RTPS)
# =====================================================================
def generate_netlist(c_val, l_val, w_switch, control_state, topology="shunt"):
    """Generates an RTPS netlist using an Ideal Branchline Coupler at 28GHz."""
    v_gate = 3.3 if control_state == 1 else 0.0
    
    # Delay for lambda/4 at 28 GHz: 1 / (4 * 28e9) = 8.928 ps
    TD = "8.928p"
    
    if topology.lower() == "shunt":
        # True Switched-LC Shunt Tank
        reflective_load_core = f"""
        * Reflective Termination Group A
        L_loadA ref_A 0 {l_val}p
        C_loadA ref_A node_swA {c_val}f
        Msw1 node_swA gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        
        * Reflective Termination Group B
        L_loadB ref_B 0 {l_val}p
        C_loadB ref_B node_swB {c_val}f
        Msw2 node_swB gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        """
    else:
        # Switched Series LC
        reflective_load_core = f"""
        * Reflective Termination Group A
        C_loadA ref_A node_swA {c_val}f
        Msw1 node_swA gate_node node_L_A 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        L_loadA node_L_A 0 {l_val}p
        
        * Reflective Termination Group B
        C_loadB ref_B node_swB {c_val}f
        Msw2 node_swB gate_node node_L_B 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        L_loadB node_L_B 0 {l_val}p
        """

    spice_deck = f"""* 28GHz RTPS Supervisory Optimization Deck
.model RF_SWITCH_MOS NMOS (LEVEL=3 TOX=4e-9 U0=600 VTO=0.7 RS=2 RD=2 CJ=1e-3 CJSW=2e-10 CGSO=3e-10 CGDO=3e-10)

Vctrl gate_node 0 DC {v_gate}

* Port 1 Input Source (50-Ohm match)
Vin input_source 0 DC 0 AC 1
Rsource input_source in 50

* =====================================================================
* IDEAL TRANSMISSION LINE BRANCHLINE COUPLER (28 GHz)
* =====================================================================
* T1: Port 1 (IN) to Port 2 (REF_A) -> Series Arm (35.35 Ohms)
T1 in 0 ref_A 0 Z0=35.35 TD={TD}
* T2: Port 1 (IN) to Port 4 (REF_B) -> Shunt Arm (50.0 Ohms)
T2 in 0 ref_B 0 Z0=50.0 TD={TD}
* T3: Port 2 (REF_A) to Port 3 (OUT) -> Shunt Arm (50.0 Ohms)
T3 ref_A 0 out 0 Z0=50.0 TD={TD}
* T4: Port 4 (REF_B) to Port 3 (OUT) -> Series Arm (35.35 Ohms)
T4 ref_B 0 out 0 Z0=35.35 TD={TD}

* =====================================================================
* TUNABLE REFLECTIVE LOAD TERMINATIONS
* =====================================================================
{reflective_load_core}

* Port 3 Output Termination (50-Ohm match)
Rload out 0 50

.ac lin 1 28g 28g
.control
    run
    wrdata s_param_output.txt vr(out) vi(out) vr(in) vi(in)
    quit
.endc
.end
"""
    return spice_deck

# =====================================================================
# 3. Simulation Runner & Parser
# =====================================================================
def run_simulation(c_val, l_val, w_switch, topology):
    results = {}
    for state in [0, 1]:
        netlist = generate_netlist(c_val, l_val, w_switch, control_state=state, topology=topology)
        
        with open("temp_deck.sp", "w") as f:
            f.write(netlist)
            
        subprocess.run([NGSPICE_EXE, "-b", "temp_deck.sp"], capture_output=True, text=True, check=True)
        
        data = np.loadtxt("s_param_output.txt")
        if data.ndim == 2:
            data = data[0]
            
        v_out_complex = data[1] + 1j * data[3]
        s21 = 2.0 * v_out_complex
        
        results[state] = {
            "loss_db": 20 * np.log10(np.abs(s21) + 1e-9),
            "phase_deg": np.degrees(np.angle(s21))
        }
        
    raw_delta = results[1]["phase_deg"] - results[0]["phase_deg"]
    phase_shift = np.abs((raw_delta + 180) % 360 - 180)
    
    worst_loss = min(results[0]["loss_db"], results[1]["loss_db"])
    return phase_shift, worst_loss

# =====================================================================
# 4. Optimization Engine Wrapper
# =====================================================================
def execute_local_synthesis(initial_seeds, target_phase, topology):
    def objective(params):
        c_val, l_val, w_switch = params
        # FIXED: Lifted bounds to prevent trapping the optimizer at 1000pH
        if c_val < 1.0 or c_val > 5000.0 or l_val < 1.0 or l_val > 5000.0 or w_switch < 5.0 or w_switch > 500.0:
            return 1e6
            
        phase_shift, loss = run_simulation(c_val, l_val, w_switch, topology)
        
        phase_error_penalty = (phase_shift - target_phase) ** 2
        loss_penalty = (abs(loss) - 6.0) ** 2 if loss < -6.0 else 0.0
        
        return (phase_error_penalty * 50.0) + loss_penalty

    # FIXED: Increased maxiter to 500 to let Nelder-Mead fully settle
    res = minimize(objective, initial_seeds, method='Nelder-Mead', options={'maxiter': 500})
    final_phase, final_loss = run_simulation(res.x[0], res.x[1], res.x[2], topology)
    
    return res, final_phase, final_loss

# =====================================================================
# 5. Main AI-Supervised Execution Loop
# =====================================================================
def main():
    target_phase = 90.0
    current_topology = "shunt" 
    
    current_seeds = [150.0, 300.0, 50.0]
    max_agent_iterations = 3
    
    last_failure_log = ""
    
    for iteration in range(1, max_agent_iterations + 1):
        print(f"\n--- 🛰️ Starting RTPS Optimization Loop [Attempt {iteration}/{max_agent_iterations}] ---")
        print(f"Reflective Load Layout: {current_topology.upper()} | Seeds: {current_seeds}")
        
        res, phase, loss = execute_local_synthesis(current_seeds, target_phase, current_topology)
        phase_error = np.abs(phase - target_phase)
        
        print(f"📊 SciPy Output -> Done (Evaluations: {res.nfev})")
        print(f"   Derived Phase Shift: {phase:.2f}° (Error: {phase_error:.2f}°)")
        print(f"   Worst-Case Insertion Loss: {loss:.2f} dB")
        
        if phase_error < 2.0 and loss > -6.0:
            print("\n🎉 RTPS Design Optimization Successful! Target metrics locked in.")
            print(f"Final Component Parameters: C={res.x[0]:.2f}fF, L={res.x[1]:.2f}pH, W={res.x[2]:.2f}um")
            return
            
        last_failure_log = f"""
        Final Reflection Topology: {current_topology}
        Phase Shift achieved: {phase:.2f}° (Target: {target_phase}°)
        Insertion Loss reached: {loss:.2f} dB
        Parameters settled on: C={res.x[0]:.2f}fF, L={res.x[1]:.2f}pH, W={res.x[2]:.2f}um
        """
        
        if iteration == max_agent_iterations:
            break
            
        print("\n⚠️ Target metrics missed. Handing diagnostic data to Gemini Design Supervisor...")
        
        SYSTEM_INSTRUCTION = """
        You are an expert Silicon Design Supervisor auditing an automated mm-wave Reflection-Type Phase Shifter (RTPS) pipeline. 
        Analyze why the local SciPy loop failed to meet its 90-degree goal at 28 GHz.
        Return an updated structured JSON coordinate decision to help the next run converge.
        Ensure component choices reflect wide bounds (L up to 5000pH, C up to 5000fF).
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Analyze this optimization log and provide corrective actions:\n{last_failure_log}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=SupervisorDecision,
                    temperature=0.3
                )
            )
            decision = json.loads(response.text)
            print(f"🤖 Gemini Supervisor Analysis: {decision['reasoning']}")
            print(f"   -> Switching Architecture to: {decision['topology'].upper()}")
            print(f"   -> Adjusting Next Starting Seeds to: [{decision['initial_c_val_ff']}, {decision['initial_l_val_ph']}, {decision['initial_w_switch_u']}]")
            
            current_topology = decision['topology']
            current_seeds = [decision['initial_c_val_ff'], decision['initial_l_val_ph'], decision['initial_w_switch_u']]
        except APIError as e:
            print(f"\n☁️ Gemini API unavailable (Status {e.code}). Applying deterministic backup recovery step...")
            current_topology = "series" if current_topology == "shunt" else "shunt"
            current_seeds = [res.x[0] * 0.8, res.x[1] * 1.2, min(res.x[2] * 1.5, 120.0)]

    print("\n❌ Optimization failed to converge within the maximum supervisor iterations.")
    print("🛰️ Generating final design post-mortem analysis...")
    
    try:
        post_mortem = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Provide a brief final engineering summary explaining what trade-offs blocked convergence given this state:\n{last_failure_log}"
        )
        print(f"\n📋 Supervisor Post-Mortem Assessment:\n{post_mortem.text}")
    except APIError:
        print("\n📋 Supervisor Post-Mortem Assessment: Cloud API is currently timed out.")

if __name__ == "__main__":
    main()

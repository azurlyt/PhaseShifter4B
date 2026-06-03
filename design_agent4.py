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
# 2. Dynamic Netlist Compiler (Fixed Bypass Topology)
# =====================================================================
def generate_netlist(c_val, l_val, w_switch, control_state, topology="shunt"):
    v_gate = 3.3 if control_state == 1 else 0.0
    TD = "8.928p" # Lambda/4 delay at 28 GHz
    
    if topology.lower() == "shunt":
        reflective_load_core = f"""
        * Reflective Termination Group A (Shunt L || Switched Shunt C)
        L_loadA ref_A 0 {l_val}p
        C_loadA ref_A node_swA {c_val}f
        Msw1 node_swA gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        
        * Reflective Termination Group B
        L_loadB ref_B 0 {l_val}p
        C_loadB ref_B node_swB {c_val}f
        Msw2 node_swB gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        """
    else:
        # FIXED: Changed from high-parasitic Series-floating to robust Series-Bypass
        # ON State: Inductor shorted out, pure series Cap to ground.
        # OFF State: Series Cap + Series Inductor tank to ground.
        reflective_load_core = f"""
        * Reflective Termination Group A (Series C + Shunt-Switched L)
        C_loadA ref_A node_midA {c_val}f
        L_loadA node_midA 0 {l_val}p
        Msw1 node_midA gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        
        * Reflective Termination Group B
        C_loadB ref_B node_midB {c_val}f
        L_loadB node_midB 0 {l_val}p
        Msw2 node_midB gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        """

    spice_deck = f"""* 28GHz RTPS Supervisory Optimization Deck
.model RF_SWITCH_MOS NMOS (LEVEL=3 TOX=4e-9 U0=600 VTO=0.7 RS=2 RD=2 CJ=1e-3 CJSW=2e-10 CGSO=3e-10 CGDO=3e-10)

Vctrl gate_node 0 DC {v_gate}

Vin input_source 0 DC 0 AC 1
Rsource input_source in 50

* 28 GHz Quadrature Hybrid Core
T1 in 0 ref_A 0 Z0=35.35 TD={TD}
T2 in 0 ref_B 0 Z0=50.0 TD={TD}
T3 ref_A 0 out 0 Z0=50.0 TD={TD}
T4 ref_B 0 out 0 Z0=35.35 TD={TD}

{reflective_load_core}

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
        if c_val < 5.0 or c_val > 1500.0 or l_val < 5.0 or l_val > 1500.0 or w_switch < 10.0 or w_switch > 300.0:
            return 1e8
            
        phase_shift, loss = run_simulation(c_val, l_val, w_switch, topology)
        
        phase_error_penalty = (phase_shift - target_phase) ** 2
        
        # FIXED: Swapped out aggressive quadratic penalty bowl for a Threshold Gate
        if loss < -4.5:
            loss_penalty = 600.0 * (abs(loss) - 4.5) ** 2
        else:
            loss_penalty = 0.0
            
        # Linear guide component to keep it scanning toward low loss
        loss_penalty += 15.0 * abs(loss)
        
        return phase_error_penalty + loss_penalty

    res = minimize(objective, initial_seeds, method='Nelder-Mead', options={'maxiter': 400})
    final_phase, final_loss = run_simulation(res.x[0], res.x[1], res.x[2], topology)
    
    return res, final_phase, final_loss

# =====================================================================
# 5. Main AI-Supervised Execution Loop
# =====================================================================
def main():
    target_phase = 180.0
    current_topology = "shunt" 
    current_seeds = [350.0, 300.0, 80.0]  # Seeding right above the shunt boundary
    max_agent_iterations = 3
    
    last_failure_log = ""
    
    for iteration in range(1, max_agent_iterations + 1):
        print(f"\n--- 🛰️ Starting RTPS Optimization Loop [Attempt {iteration}/{max_agent_iterations}] ---")
        print(f"Reflective Load Layout: {current_topology.upper()} | Seeds: {[round(x,1) for x in current_seeds]}")
        
        res, phase, loss = execute_local_synthesis(current_seeds, target_phase, current_topology)
        phase_error = np.abs(phase - target_phase)
        
        print(f"📊 SciPy Output -> Done (Evaluations: {res.nfev})")
        print(f"   Derived Phase Shift: {phase:.2f}° (Error: {phase_error:.2f}°)")
        print(f"   Worst-Case Insertion Loss: {loss:.2f} dB")
        
        if phase_error < 5.0 and loss > -5.0:
            print("\n🎉 RTPS 180° Design Optimization Successful! All stages complete.")
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
        
        # FIXED: Added explicit microwave boundary constraints to the agent's guide context
        SYSTEM_INSTRUCTION = """
        You are an expert Silicon Design Supervisor auditing an automated mm-wave 180-degree RTPS pipeline at 28 GHz.
        Component boundaries must stay below 1500pH and 1500fF. 
        
        Physics Guide Guidelines:
        1. If choosing 'shunt' topology, the required capacitance for a 180-deg swing physically requires C > 227 fF. 
        2. If choosing 'bypass' topology, the design exhibits an elegant symmetric matching target around C = 114 fF and L = 568 pH.
        
        Suggest appropriate architectural switches or updated scaling parameters to let the optimizer lock targets.
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
            print(f"   -> Adjusting Next Starting Seeds to: [{decision['initial_c_val_ff']:.1f}, {decision['initial_l_val_ph']:.1f}, {decision['initial_w_switch_u']:.1f}]")
            
            current_topology = decision['topology']
            current_seeds = [decision['initial_c_val_ff'], decision['initial_l_val_ph'], decision['initial_w_switch_u']]
        except APIError as e:
            print(f"\n☁️ Gemini API unavailable (Status {e.code}). Applying deterministic backup recovery step...")
            current_topology = "bypass" if current_topology == "shunt" else "shunt"
            current_seeds = [114.0, 568.0, 100.0]

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

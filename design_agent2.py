import os
import subprocess
import json
import numpy as np
from scipy.optimize import minimize
from pydantic import BaseModel
from google import genai
from google.genai import types

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
# 2. Dynamic Netlist Compiler (FIXED: True Switched-Shunt Pi Topology)
# =====================================================================
def generate_netlist(c_val, l_val, w_switch, control_state, topology="pi"):
    v_gate = 3.3 if control_state == 1 else 0.0
    
    if topology.lower() == "t":
        filter_core = f"""
        L1 in node_a {l_val/2}p
        L2 node_a out {l_val/2}p
        Msw1 node_a gate_node node_sw 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        C1 node_sw 0 {c_val}f
        """
    else:
        # FIXED: Inductor stays in series. Dual shunt capacitors are switched to ground.
        filter_core = f"""
        L1 in out {l_val}p
        C1 in node_sw1 {c_val}f
        Msw1 node_sw1 gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        C2 out node_sw2 {c_val}f
        Msw2 node_sw2 gate_node 0 0 RF_SWITCH_MOS W={w_switch}u L=0.45u
        """

    spice_deck = f"""* 28GHz Supervisory Optimization Deck
        .model RF_SWITCH_MOS NMOS (LEVEL=3 TOX=4e-9 U0=600 VTO=0.7 RS=5 RD=5 CJ=1e-3 CJSW=2e-10 CGSO=3e-10 CGDO=3e-10)

        Vctrl gate_node 0 DC {v_gate}

        Vin input_source 0 DC 0 AC 1
        Rsource input_source in 50

        {filter_core}

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
# 3. Simulation Runner & Parser (FIXED: Monitored Both States)
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
        
    # FIXED: Handled 360-degree wrapping for accurate relative delta phase
    raw_delta = results[1]["phase_deg"] - results[0]["phase_deg"]
    phase_shift = np.abs((raw_delta + 180) % 360 - 180)
    
    # FIXED: Track worst-case insertion loss across BOTH states to eliminate open-circuit exploit
    worst_loss = min(results[0]["loss_db"], results[1]["loss_db"])
    
    return phase_shift, worst_loss

# =====================================================================
# 4. Optimization Engine Wrapper (FIXED: Objective Function Weights)
# =====================================================================
def execute_local_synthesis(initial_seeds, target_phase, topology):
    def objective(params):
        c_val, l_val, w_switch = params
        
        # Hard limits to keep components in realistic physics boundaries
        if c_val < 5.0 or c_val > 1000.0 or l_val < 5.0 or l_val > 2000.0 or w_switch < 5.0 or w_switch > 400.0:
            return 1e6
            
        phase_shift, worst_loss = run_simulation(c_val, l_val, w_switch, topology)
        
        # INCREASED: Make phase accuracy the absolute top priority
        phase_error_penalty = 100.0 * (phase_shift - target_phase) ** 2
        
        # Hard barrier if transmission drops below -3.0 dB
        loss_penalty = 0.0
        if worst_loss < -3.0:
            loss_penalty = 500.0 * (worst_loss + 3.0) ** 2
            
        # DECREASED: Make general transparency a gentle secondary nudge
        smooth_loss_penalty = 1.0 * (worst_loss) ** 2
        
        return phase_error_penalty + loss_penalty + smooth_loss_penalty

    # Increased maxiter to 200 to give it room to crawl down the new gradient
    res = minimize(objective, initial_seeds, method='Nelder-Mead', options={'maxiter': 200})
    final_phase, final_loss = run_simulation(res.x[0], res.x[1], res.x[2], topology)
    
    return res, final_phase, final_loss

# =====================================================================
# 5. Main AI-Supervised Execution Loop
# =====================================================================
def main():
    target_phase = 45.0  # Can change to 22.5 safely
    current_topology = "pi"
    
    # FIXED: Re-seeded optimizer closer to the physical sweet spot for 28 GHz loading lines
    current_seeds = [100.0, 230.0, 100.0]
    max_agent_iterations = 3
    
    last_failure_log = ""
    
    for iteration in range(1, max_agent_iterations + 1):
        print(f"\n--- 🛰️ Starting Optimization Loop [Attempt {iteration}/{max_agent_iterations}] ---")
        print(f"Topology: {current_topology.upper()} | Starting Seeds: {current_seeds}")
        
        res, phase, loss = execute_local_synthesis(current_seeds, target_phase, current_topology)
        phase_error = np.abs(phase - target_phase)
        
        print(f"📊 SciPy Output -> Done (Evaluations: {res.nfev})")
        print(f"   Derived Phase Shift: {phase:.2f}° (Error: {phase_error:.2f}°)")
        print(f"   Worst-Case Insertion Loss: {loss:.2f} dB")
        
        # Valid engineering tolerances (Pass if phase error < 1.0 deg and transmission > -3 dB)
        if phase_error < 1.0 and loss > -3.0:
            print("\n🎉 Design Optimization Successful! Target metrics locked in.")
            print(f"Final Component Parameters: C={res.x[0]:.2f}fF, L={res.x[1]:.2f}pH, W={res.x[2]:.2f}um")
            return
            
        last_failure_log = f"""
        Final Topology: {current_topology}
        Phase Shift achieved: {phase:.2f}° (Target: {target_phase}°)
        Worst-Case Insertion Loss reached: {loss:.2f} dB
        Parameters settled on: C={res.x[0]:.2f}fF, L={res.x[1]:.2f}pH, W={res.x[2]:.2f}um
        """
        
        if iteration == max_agent_iterations:
            break
            
        print("\n⚠️ Target metrics missed. Handing diagnostic data to Gemini Design Supervisor...")
        
        SYSTEM_INSTRUCTION = """
        You are an expert Silicon Design Supervisor auditing an automated mm-wave IC optimization pipeline. 
        Your task is to analyze why a local SciPy optimization failed to hit its performance goals at 28 GHz.
        Provide updated starting seeds or flip the topology to escape local minima.
        """
        
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

    print("\n❌ Optimization failed to converge within the maximum supervisor iterations.")
    print("🛰️ Generating final design post-mortem analysis with Gemini...")
    
    post_mortem = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Provide a brief final engineering summary explaining what trade-offs blocked convergence given this state:\n{last_failure_log}"
    )
    print(f"\n📋 Supervisor Post-Mortem Assessment:\n{post_mortem.text}")

if __name__ == "__main__":
    main()

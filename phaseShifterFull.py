import os
import subprocess
import numpy as np

# =====================================================================
# 1. Path Configurations
# =====================================================================
NGSPICE_EXE = r"C:\Spice64\bin\ngspice.exe"

# =====================================================================
# 2. Optimized Stage Parameter Database
# =====================================================================
STAGES_CONFIG = {
    "bit_22_5": {"C": "62.72f",   "L": "237.73p",  "W": "87.45u"},
    "bit_45":   {"C": "111.63f",  "L": "298.23p", "W": "86.27u"},
    "bit_90":   {"C": "41.36f",  "L": "1239.82p","W": "38.28u"}  
}

# =====================================================================
# 3. Comprehensive Netlist Generator with Buffers
# =====================================================================
def generate_full_shifter_netlist(state_vector):
    # The external bitstream stays entirely unchanged (Active-High)
    v_g180 = 3.3 if state_vector[0] == 1 else 0.0
    v_g90  = 3.3 if state_vector[1] == 1 else 0.0
    
    # The user passes active-high, but we silently invert the lower bits 
    # internally so the SPICE engine triggers the low-pass stages properly.
    v_g45  = 0.0 if state_vector[2] == 1 else 3.3
    v_g22_5= 0.0 if state_vector[3] == 1 else 3.3

    TD_28G = "8.928p"

    spice_deck = f"""* Full 4-Bit Cascaded 28GHz Phase Shifter Simulation Deck

* RF Switch MOS Model Definition
.model RF_SWITCH_MOS NMOS (LEVEL=3 TOX=4e-9 U0=600 VTO=0.7 RS=2 RD=2 CJ=1e-3 CJSW=2e-10 CGSO=3e-10 CGDO=3e-10)

* Control Voltage Rails
V_ctrl_180  g180  0 DC {v_g180}
V_ctrl_90   g90   0 DC {v_g90}
V_ctrl_45   g45   0 DC {v_g45}
V_ctrl_22_5 g22_5 0 DC {v_g22_5}

* Port 1 Main RF Input Source
Vin input_node 0 DC 0 AC 1
Rsource input_node in 50

* =====================================================================
* STAGE 1: 22.5 Degree Pi Switched-Filter Stage
* =====================================================================
X_stage22_5 in n_out_22 g22_5 stage_pi_22_5

* BUFFER 1 (Isolates 22.5 deg stage from 45 deg stage)
R_term_22 n_out_22 0 50
E_buff_1 n_src_45 0 n_out_22 0 2.0
R_src_45 n_src_45 n_in_45 50

* =====================================================================
* STAGE 2: 45 Degree Pi Switched-Filter Stage
* =====================================================================
X_stage45 n_in_45 n_out_45 g45 stage_pi_45

* BUFFER 2 (Isolates 45 deg stage from 90 deg stage)
R_term_45 n_out_45 0 50
E_buff_2 n_src_90 0 n_out_45 0 2.0
R_src_90 n_src_90 n_in_90 50

* =====================================================================
* STAGE 3: 90 Degree RTPS Stage
* =====================================================================
X_stage90 n_in_90 n_out_90 g90 stage_rtps_90

* BUFFER 3 (Isolates 90 deg stage from first half of 180 deg stage)
R_term_90 n_out_90 0 50
E_buff_3 n_src_180a 0 n_out_90 0 2.0
R_src_180a n_src_180a n_in_180a 50

* =====================================================================
* STAGE 4: 180 Degree Stage (Cascaded 2x 90 Degree RTPS Blocks)
* =====================================================================
X_stage180_part1 n_in_180a n_out_180a g180 stage_rtps_90

* BUFFER 4 (Isolates the two 90 deg blocks)
R_term_180a n_out_180a 0 50
E_buff_4 n_src_180b 0 n_out_180a 0 2.0
R_src_180b n_src_180b n_in_180b 50

X_stage180_part2 n_in_180b n_out_final g180 stage_rtps_90

* Port 2 Main RF Output Termination
Rload n_out_final 0 50

* =====================================================================
* SUBCIRCUIT DEFINITIONS
* =====================================================================

* --- 22.5 Degree Switched Pi Subcircuit ---
.subckt stage_pi_22_5 node_in node_out gate_node
L_ser node_in node_out {STAGES_CONFIG['bit_22_5']['L']}
C_sh1 node_in node_sw1 {STAGES_CONFIG['bit_22_5']['C']}
C_sh2 node_out node_sw2 {STAGES_CONFIG['bit_22_5']['C']}
Msw1 node_sw1 gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_22_5']['W']} L=0.45u
Msw2 node_sw2 gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_22_5']['W']} L=0.45u
.ends

* --- 45 Degree Switched Pi Subcircuit ---
.subckt stage_pi_45 node_in node_out gate_node
L_ser node_in node_out {STAGES_CONFIG['bit_45']['L']}
C_sh1 node_in node_sw1 {STAGES_CONFIG['bit_45']['C']}
C_sh2 node_out node_sw2 {STAGES_CONFIG['bit_45']['C']}
Msw1 node_sw1 gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_45']['W']} L=0.45u
Msw2 node_sw2 gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_45']['W']} L=0.45u
.ends

* --- 90 Degree Reflective Type Phase Shifter Subcircuit ---
.subckt stage_rtps_90 node_in node_out gate_node
T1 node_in 0 ref_A  0 Z0=35.35 TD={TD_28G}
T2 node_in 0 ref_B  0 Z0=50.0  TD={TD_28G}
T3 ref_A   0 node_out 0 Z0=50.0  TD={TD_28G}
T4 ref_B   0 node_out 0 Z0=35.35 TD={TD_28G}

L_loadA ref_A 0 {STAGES_CONFIG['bit_90']['L']}
C_loadA ref_A node_swA {STAGES_CONFIG['bit_90']['C']}
MswA node_swA gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_90']['W']} L=0.45u

L_loadB ref_B 0 {STAGES_CONFIG['bit_90']['L']}
C_loadB ref_B node_swB {STAGES_CONFIG['bit_90']['C']}
MswB node_swB gate_node 0 0 RF_SWITCH_MOS W={STAGES_CONFIG['bit_90']['W']} L=0.45u
.ends

* Simulation Directives
.ac lin 1 28g 28g
.control
    run
    wrdata system_cascade_out.txt vr(n_out_final) vi(n_out_final) vr(in) vi(in)
    quit
.endc
.end
"""
    return spice_deck

# =====================================================================
# 4. Simulation Engine & 16-State Sweeper
# =====================================================================
def run_state_simulation(state):
    deck = generate_full_shifter_netlist(state)
    with open("full_shifter_run.sp", "w") as f:
        f.write(deck)
        
    subprocess.run([NGSPICE_EXE, "-b", "full_shifter_run.sp"], capture_output=True, text=True, check=True)
    
    raw_data = np.loadtxt("system_cascade_out.txt")
    if raw_data.ndim == 2:
        raw_data = raw_data[0]
        
    # S21 Insertion Loss & Phase 
    v_out_complex = raw_data[1] + 1j * raw_data[3]
    s21 = 2.0 * v_out_complex
    loss_db = 20 * np.log10(np.abs(s21) + 1e-9)
    phase_deg = np.degrees(np.angle(s21))
    
    # S11 Return Loss Calculation
    v_in_complex = raw_data[4] + 1j * raw_data[6]
    s11 = 2.0 * v_in_complex - 1.0  
    return_loss_db = 20 * np.log10(np.abs(s11) + 1e-9)
    
    return loss_db, phase_deg, return_loss_db

def main():
    print("🛰️ Initializing Buffered 4-Bit Cascaded Phase Shifter Simulation...")
    
    results_table = []
    base_phase = None
    
    # The tracking loop counts up completely normally from 0000 to 1111
    for b180 in [0, 1]:
        for b90 in [0, 1]:
            for b45 in [0, 1]:
                for b22_5 in [0, 1]:
                    state = [b180, b90, b45, b22_5]
                    state_str = "".join(map(str, state))
                    
                    expected_shift = (b180 * 180.0) + (b90 * 90.0) + (b45 * 45.0) + (b22_5 * 22.5)
                    
                    loss, raw_phase, return_loss = run_state_simulation(state)
                    
                    if state_str == "0000":
                        base_phase = raw_phase
                        relative_phase = 0.0
                    else:
                        raw_delta = raw_phase - base_phase
                        relative_phase = (raw_delta + 180) % 360 - 180
                        if relative_phase < -10: 
                            relative_phase += 360
                            
                    phase_error = relative_phase - expected_shift
                    
                    # Correct multi-wrap boundary edge cases near 360° for clean reporting
                    if phase_error > 180:  phase_error -= 360
                    if phase_error < -180: phase_error += 360
                        
                    results_table.append({
                        "state": state_str,
                        "expected": expected_shift,
                        "simulated": relative_phase,
                        "error": phase_error,
                        "loss": loss,
                        "return_loss": return_loss
                    })

    print("\n### 📊 4-Bit Buffered System Simulation Matrix (28 GHz)")
    print("| Digital State | Target Shift (°) | Simulated Shift (°) | Phase Error (°) | Insertion Loss (dB) | Return Loss (dB) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    
    losses = []
    return_losses = []
    errors = []
    for r in results_table:
        print(f"| {r['state']} | {r['expected']:6.1f}° | {r['simulated']:6.1f}° | {r['error']:+6.2f}° | {r['loss']:6.2f} dB | {r['return_loss']:6.2f} dB |")
        losses.append(r['loss'])
        return_losses.append(r['return_loss'])
        if r['state'] != "0000":
            errors.append(abs(r['error']))
            
    print("\n### 📈 Integrated Performance Summary")
    print(f"* **Average Insertion Loss:** {np.mean(losses):.2f} dB")
    print(f"* **Worst-Case Insertion Loss:** {min(losses):.2f} dB")
    print(f"* **Worst-Case Return Loss (S11):** {max(return_losses):.2f} dB")
    print(f"* **Peak Phase Error:** {max(errors):.2f}°")
    print(f"* **RMS Phase Error:** {np.sqrt(np.mean(np.array(errors)**2)):.2f}°")

if __name__ == "__main__":
    main()
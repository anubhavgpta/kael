# -----------------------------------------------------------------------------
# Script : setup_project.tcl
# Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
# Purpose: Add all RTL and TB sources to existing Vivado project
# Usage  : Open Vivado project kael.xpr, then in Tcl console:
#          source C:/Users/Anubhav\ Gupta/Desktop/Projects/kael/vivado/setup_project.tcl
# -----------------------------------------------------------------------------

set RTL_DIR [file normalize "C:/Users/Anubhav Gupta/Desktop/Projects/kael/rtl"]
set TB_DIR  [file normalize "C:/Users/Anubhav Gupta/Desktop/Projects/kael/tb"]

# -----------------------------------------------------------------------------
# RTL sources
# -----------------------------------------------------------------------------
add_files -fileset sources_1 [list \
    $RTL_DIR/qk_dot_engine.v  \
    $RTL_DIR/score_scaler.v   \
    $RTL_DIR/softmax_engine.v \
    $RTL_DIR/v_accumulator.v  \
    $RTL_DIR/attention_ctrl.v \
]

set_property file_type SystemVerilog [get_files $RTL_DIR/qk_dot_engine.v]
set_property file_type SystemVerilog [get_files $RTL_DIR/score_scaler.v]
set_property file_type SystemVerilog [get_files $RTL_DIR/softmax_engine.v]
set_property file_type SystemVerilog [get_files $RTL_DIR/v_accumulator.v]
set_property file_type SystemVerilog [get_files $RTL_DIR/attention_ctrl.v]

# -----------------------------------------------------------------------------
# LUT memory init file
# -----------------------------------------------------------------------------
add_files -fileset sources_1 $RTL_DIR/exp_lut.mem

# -----------------------------------------------------------------------------
# Simulation sources
# -----------------------------------------------------------------------------
add_files -fileset sim_1 [list \
    $TB_DIR/tb_qk_dot_engine.sv  \
    $TB_DIR/tb_score_scaler.sv   \
    $TB_DIR/tb_softmax_engine.sv \
    $TB_DIR/tb_v_accumulator.sv  \
    $TB_DIR/tb_attention_ctrl.sv \
]

set_property file_type SystemVerilog [get_files $TB_DIR/tb_qk_dot_engine.sv]
set_property file_type SystemVerilog [get_files $TB_DIR/tb_score_scaler.sv]
set_property file_type SystemVerilog [get_files $TB_DIR/tb_softmax_engine.sv]
set_property file_type SystemVerilog [get_files $TB_DIR/tb_v_accumulator.sv]
set_property file_type SystemVerilog [get_files $TB_DIR/tb_attention_ctrl.sv]

# -----------------------------------------------------------------------------
# Set tops
# -----------------------------------------------------------------------------
set_property top attention_ctrl   [get_fileset sources_1]
set_property top tb_attention_ctrl [get_fileset sim_1]

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
puts ""
puts "Sources loaded into project kael"
puts "RTL  : qk_dot_engine, score_scaler, softmax_engine, v_accumulator, attention_ctrl"
puts "TB   : tb_qk_dot_engine, tb_score_scaler, tb_softmax_engine,"
puts "       tb_v_accumulator, tb_attention_ctrl"
puts "LUT  : exp_lut.mem"
puts "Synth top : attention_ctrl"
puts "Sim top   : tb_attention_ctrl"
puts ""
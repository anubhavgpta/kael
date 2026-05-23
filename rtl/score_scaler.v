`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Module : score_scaler
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: Scale dot product by 1/sqrt(64)=0.125 via right-shift, Q16.16->Q1.15
// -----------------------------------------------------------------------------
//
// Scaling math
// ------------
// HEAD_DIM = 64  =>  1/sqrt(64) = 1/8  (exact power of two, zero DSP cost)
// Dividing a Q16.16 value by 8 is an arithmetic right-shift of 3 positions.
//
// Bit extraction
// --------------
// dot_in is Q16.16 (32-bit signed):
//   bit[31]    = sign
//   bits[30:16] = integer part  (decimal point between bit[15] and bit[16])
//   bits[15:0]  = fractional part
//
// Q1.15_raw = Q16.16_raw / 2^15 * 2^15 / 8 = Q16.16_raw >> 4
// So the Q1.15 raw integer is dot_in >> 4, and its lower 16 bits are dot_in[19:4]:
//   bit[19] of dot_in becomes the sign of the 16-bit Q1.15 result
//   bits[18:4] of dot_in are the 15 fractional bits below it
// => extract dot_in[19:4] directly -- no explicit shifter needed.
//
// Saturation
// ----------
// No overflow <=> bits[31:20] of dot_in are all equal to bit[19]
//   (12-bit sign-extension check for the window covered by dot_in[19:4])
// Overflow positive (dot_in[31]=0): clamp to 16'h7FFF
// Overflow negative (dot_in[31]=1): clamp to 16'h8000
//
// Pipeline
// --------
// One register stage: {dot_in, dot_valid} -> registered {score_out, score_valid}
// Latency = 1 cycle.
// -----------------------------------------------------------------------------
module score_scaler #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 16
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [IN_WIDTH-1:0]   dot_in,    // Q16.16 from qk_dot_engine
    input  logic                  dot_valid,  // qualifies dot_in
    output logic [OUT_WIDTH-1:0]  score_out,  // Q1.15 scaled attention score
    output logic                  score_valid  // qualifies score_out
);

    // -------------------------------------------------------------------------
    // Overflow detection: bits[31:20] must all replicate bit[19]
    // -------------------------------------------------------------------------
    // The 12 bits above bit[19] must all be copies of the Q1.15 sign.
    // Any deviation means the value overflows the [-8, 8) window captured
    // by the dot_in[19:4] slice.
    logic overflow; // asserted when dot_in[19:4] would misrepresent the value
    assign overflow = (dot_in[IN_WIDTH-1:20] != {12{dot_in[19]}});

    // -------------------------------------------------------------------------
    // Combinatorial scaled value with saturation
    // -------------------------------------------------------------------------
    logic [OUT_WIDTH-1:0] scaled_comb;

    always_comb begin
        if (overflow)
            // dot_in[IN_WIDTH-1] is the true sign of the Q16.16 number
            scaled_comb = dot_in[IN_WIDTH-1] ? 16'h8000 : 16'h7FFF;
        else
            // Shift-by-4 implemented as a slice: bit[19] -> Q1.15 sign,
            // bits[18:4] -> 15 fractional bits.
            scaled_comb = dot_in[19:4];
    end

    // -------------------------------------------------------------------------
    // Pipeline register: one cycle latency
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_out   <= '0;
            score_valid <= 1'b0;
        end else begin
            score_valid <= dot_valid;
            if (dot_valid)
                score_out <= scaled_comb;
        end
    end

endmodule

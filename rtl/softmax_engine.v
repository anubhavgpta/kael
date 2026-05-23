`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Module : softmax_engine
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: Online streaming softmax, running max + rescale, Flash Attention style
// -----------------------------------------------------------------------------
//
// Maintains the online softmax denominator:
//   m_reg = running maximum score, Q1.15 signed
//   d_reg = running sum of exp(score - m_reg), Q16.16 signed
//
// Pipeline
// --------
// Cycle N   : accept score, compute m_new and LUT indices, update m_reg
// Cycle N+1 : register LUT outputs
// Cycle N+2 : update d_reg and emit weight/rescale/denominator valids
// -----------------------------------------------------------------------------
module softmax_engine #(
    parameter SCORE_WIDTH = 16,
    parameter ACC_WIDTH   = 32,
    parameter EXP_WIDTH   = 16
)(
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic [SCORE_WIDTH-1:0]   score_in,
    input  logic                     score_valid,
    input  logic                     seq_last,    // pulses with last score_valid of sequence
    input  logic                     seq_start,   // pulse before first score of new sequence

    // To v_accumulator
    output logic [EXP_WIDTH-1:0]     weight_out,
    output logic                     weight_valid,
    output logic                     rescale_valid,
    output logic [EXP_WIDTH-1:0]     rescale_factor,

    // Final denominator for context vector normalization
    output logic [ACC_WIDTH-1:0]     denom_out,
    output logic                     denom_valid
);

    // -------------------------------------------------------------------------
    // Constants and LUT storage
    // -------------------------------------------------------------------------
    localparam [SCORE_WIDTH-1:0] MIN_SCORE = 16'h8000;
    localparam [SCORE_WIDTH-1:0] LUT_SHIFT = 16'h7FFF;
    localparam signed [SCORE_WIDTH:0] MIN_DIFF = {1'b1, MIN_SCORE};

    logic [EXP_WIDTH-1:0] exp_lut [0:255];

    initial begin
        $readmemh("rtl/exp_lut.mem", exp_lut);
    end

    // -------------------------------------------------------------------------
    // Online softmax state
    // -------------------------------------------------------------------------
    logic signed [SCORE_WIDTH-1:0] m_reg; // Q1.15 running maximum
    logic signed [ACC_WIDTH-1:0]   d_reg; // Q16.16 denominator accumulator
    logic                          seen_score;

    // -------------------------------------------------------------------------
    // Stage 1 -> Stage 2: LUT request metadata
    // -------------------------------------------------------------------------
    logic                         s1_valid;
    logic                         s1_new_max;
    logic                         s1_last;
    logic [7:0]                   s1_weight_idx;
    logic [7:0]                   s1_rescale_idx;

    // -------------------------------------------------------------------------
    // Stage 2 -> Stage 3: registered LUT outputs and metadata
    // -------------------------------------------------------------------------
    logic                         s2_valid;
    logic                         s2_new_max;
    logic                         s2_last;
    logic [EXP_WIDTH-1:0]         s2_weight_exp;
    logic [EXP_WIDTH-1:0]         s2_rescale_exp;

    // -------------------------------------------------------------------------
    // Temporary arithmetic values used inside registered pipeline blocks
    // -------------------------------------------------------------------------
    logic signed [SCORE_WIDTH-1:0] score_s;
    logic signed [SCORE_WIDTH-1:0] m_new;
    logic signed [SCORE_WIDTH:0]   weight_diff;
    logic signed [SCORE_WIDTH:0]   rescale_diff;
    logic signed [SCORE_WIDTH-1:0] weight_diff_sat;
    logic signed [SCORE_WIDTH-1:0] rescale_diff_sat;
    logic [SCORE_WIDTH:0]          weight_shifted;
    logic [SCORE_WIDTH:0]          rescale_shifted;
    logic [24:0]                   weight_scaled;
    logic [24:0]                   rescale_scaled;
    logic [ACC_WIDTH+EXP_WIDTH-1:0] d_rescale_product;
    logic [ACC_WIDTH-1:0]          d_rescaled;
    logic [ACC_WIDTH-1:0]          d_next;

    // -------------------------------------------------------------------------
    // Pipeline stage 1: compare against running max and form LUT indices
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_reg          <= MIN_SCORE;
            seen_score     <= 1'b0;
            s1_valid       <= 1'b0;
            s1_new_max     <= 1'b0;
            s1_last        <= 1'b0;
            s1_weight_idx  <= '0;
            s1_rescale_idx <= '0;
        end else if (seq_start) begin
            m_reg          <= MIN_SCORE;
            seen_score     <= 1'b0;
            s1_valid       <= 1'b0;
            s1_new_max     <= 1'b0;
            s1_last        <= 1'b0;
            s1_weight_idx  <= '0;
            s1_rescale_idx <= '0;
        end else begin
            s1_valid <= score_valid;
            s1_last  <= score_valid && seq_last;

            if (score_valid) begin
                score_s = score_in;
                m_new   = (!seen_score || (score_s > m_reg)) ? score_s : m_reg;

                weight_diff  = {score_s[SCORE_WIDTH-1], score_s} -
                               {m_new[SCORE_WIDTH-1], m_new};
                rescale_diff = {m_reg[SCORE_WIDTH-1], m_reg} -
                               {m_new[SCORE_WIDTH-1], m_new};

                weight_diff_sat  = (weight_diff < MIN_DIFF) ?
                                   MIN_SCORE : weight_diff[SCORE_WIDTH-1:0];
                rescale_diff_sat = (rescale_diff < MIN_DIFF) ?
                                   MIN_SCORE : rescale_diff[SCORE_WIDTH-1:0];

                weight_shifted  = $signed({weight_diff_sat[SCORE_WIDTH-1], weight_diff_sat}) +
                                  $signed({1'b0, LUT_SHIFT});
                rescale_shifted = $signed({rescale_diff_sat[SCORE_WIDTH-1], rescale_diff_sat}) +
                                  $signed({1'b0, LUT_SHIFT});
                weight_scaled   = weight_shifted * 8'd255;
                rescale_scaled  = rescale_shifted * 8'd255;

                s1_weight_idx  <= (weight_diff_sat >= 0) ? 8'hFF :
                                  (weight_diff_sat == $signed(MIN_SCORE)) ? 8'h00 :
                                  weight_scaled[22:15];
                s1_rescale_idx <= (rescale_diff_sat >= 0) ? 8'hFF :
                                  (rescale_diff_sat == $signed(MIN_SCORE)) ? 8'h00 :
                                  rescale_scaled[22:15];
                s1_new_max     <= seen_score && (m_new != m_reg);
                m_reg          <= m_new;
                seen_score     <= 1'b1;
            end else begin
                s1_new_max     <= 1'b0;
                s1_weight_idx  <= '0;
                s1_rescale_idx <= '0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Pipeline stage 2: registered LUT lookup
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid       <= 1'b0;
            s2_new_max     <= 1'b0;
            s2_last        <= 1'b0;
            s2_weight_exp  <= '0;
            s2_rescale_exp <= '0;
        end else if (seq_start) begin
            s2_valid   <= 1'b0;
            s2_new_max <= 1'b0;
            s2_last    <= 1'b0;
        end else begin
            s2_valid   <= s1_valid;
            s2_new_max <= s1_new_max;
            s2_last    <= s1_last;

            if (s1_valid) begin
                s2_weight_exp  <= exp_lut[s1_weight_idx];
                s2_rescale_exp <= exp_lut[s1_rescale_idx];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Pipeline stage 3: update denominator and emit aligned outputs
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d_reg          <= '0;
            weight_out     <= '0;
            weight_valid   <= 1'b0;
            rescale_valid  <= 1'b0;
            rescale_factor <= '0;
            denom_out      <= '0;
            denom_valid    <= 1'b0;
        end else if (seq_start) begin
            d_reg         <= '0;
            weight_valid  <= 1'b0;
            rescale_valid <= 1'b0;
            denom_valid   <= 1'b0;
        end else begin
            weight_valid  <= s2_valid;
            rescale_valid <= s2_valid && s2_new_max;
            denom_valid   <= s2_valid && s2_last;

            if (s2_valid) begin
                d_rescale_product = d_reg * s2_rescale_exp;
                d_rescaled        = d_rescale_product[ACC_WIDTH+14:15];
                d_next            = (s2_new_max ? d_rescaled : d_reg) +
                                    {{(ACC_WIDTH-EXP_WIDTH){1'b0}}, s2_weight_exp};

                d_reg          <= d_next;
                weight_out     <= s2_weight_exp;
                rescale_factor <= s2_rescale_exp;

                if (s2_last)
                    denom_out <= d_next;
            end
        end
    end

endmodule

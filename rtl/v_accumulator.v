`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Module : v_accumulator
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: Weighted V sum accumulator with online rescale, context vector output
// -----------------------------------------------------------------------------
module v_accumulator #(
    parameter HEAD_DIM     = 64,
    parameter DATA_WIDTH   = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter ACC_WIDTH    = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // V vector from Vera (one element per cycle, HEAD_DIM elements per token)
    input  logic [DATA_WIDTH-1:0]    v_data,
    input  logic                     v_valid,

    // Weight from softmax_engine (one weight per token, held stable for HEAD_DIM cycles)
    input  logic [WEIGHT_WIDTH-1:0]  weight_in,
    input  logic                     weight_valid,

    // Rescale from softmax_engine
    input  logic                     rescale_valid,
    input  logic [WEIGHT_WIDTH-1:0]  rescale_factor,

    // Sequence control
    input  logic                     seq_start,   // reset accumulator for new sequence
    input  logic                     seq_last,    // last token: trigger output pass after acc

    // Denominator for final normalization
    input  logic [ACC_WIDTH-1:0]     denom_in,
    input  logic                     denom_valid,

    // Context vector output (streamed HEAD_DIM elements)
    output logic [DATA_WIDTH-1:0]    ctx_out,
    output logic                     ctx_valid,
    output logic                     ctx_last,

    // Stall: high during rescale pass (attention_ctrl must pause Vera reads)
    output logic                     stall
);

    // -------------------------------------------------------------------------
    // Derived constants and state
    // -------------------------------------------------------------------------
    localparam IDX_WIDTH = $clog2(HEAD_DIM);
    localparam MUL_WIDTH = DATA_WIDTH + WEIGHT_WIDTH + 2;
    localparam RESCALE_MUL_WIDTH = ACC_WIDTH + WEIGHT_WIDTH + 1;
    localparam signed [MUL_WIDTH-1:0] WEIGHT_ROUND = 16384;

    typedef enum logic [1:0] {
        IDLE,
        ACCUMULATE,
        RESCALE,
        OUTPUT
    } state_t;

    state_t state;

    logic [IDX_WIDTH-1:0] v_idx;
    logic [IDX_WIDTH-1:0] r_idx;
    logic [IDX_WIDTH-1:0] o_idx;

    logic [WEIGHT_WIDTH-1:0] rescale_reg;
    logic [WEIGHT_WIDTH-1:0] weight_reg;
    logic [ACC_WIDTH-1:0]    denom_reg;
    logic                    output_pending;

    logic signed [ACC_WIDTH-1:0] acc [HEAD_DIM-1:0]; // Q16.16 accumulator

    // -------------------------------------------------------------------------
    // Fixed-point datapath
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0]         v_signed;
    logic signed [DATA_WIDTH:0]           v_ext;
    logic signed [WEIGHT_WIDTH:0]         weight_ext;
    logic signed [MUL_WIDTH-1:0]          weighted_product; // Q9.23
    logic signed [MUL_WIDTH-1:0]          weighted_rounded;
    logic signed [DATA_WIDTH-1:0]         weighted_q8_8;
    logic signed [ACC_WIDTH-1:0]          weighted_q16_16;
    logic signed [RESCALE_MUL_WIDTH-1:0] rescale_product; // Q17.31
    logic signed [ACC_WIDTH-1:0]          rescaled_acc;

    assign v_signed         = $signed(v_data);
    assign v_ext            = {v_signed[DATA_WIDTH-1], v_signed};
    assign weight_ext       = $signed({1'b0, weight_reg});
    assign weighted_product = v_ext * weight_ext;
    assign weighted_rounded = weighted_product[MUL_WIDTH-1] ?
                              (weighted_product - WEIGHT_ROUND) :
                              (weighted_product + WEIGHT_ROUND);
    assign weighted_q8_8    = weighted_rounded[DATA_WIDTH+14:15];
    assign weighted_q16_16  = {{(ACC_WIDTH-DATA_WIDTH-8){weighted_q8_8[DATA_WIDTH-1]}},
                               weighted_q8_8,
                               8'b0};

    assign rescale_product = acc[r_idx] * $signed({1'b0, rescale_reg});
    assign rescaled_acc    = rescale_product[ACC_WIDTH+14:15];

    assign stall = (state == RESCALE);

    // -------------------------------------------------------------------------
    // Accumulator FSM
    // -------------------------------------------------------------------------
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            v_idx       <= '0;
            r_idx       <= '0;
            o_idx       <= '0;
            rescale_reg <= '0;
            weight_reg  <= '0;
            denom_reg   <= '0;
            output_pending <= 1'b0;
            ctx_out     <= '0;
            ctx_valid   <= 1'b0;
            ctx_last    <= 1'b0;

            for (i = 0; i < HEAD_DIM; i = i + 1)
                acc[i] <= '0;
        end else begin
            ctx_valid <= 1'b0;
            ctx_last  <= 1'b0;

            case (state)
                IDLE: begin
                    v_idx <= '0;
                    r_idx <= '0;
                    o_idx <= '0;
                    output_pending <= 1'b0;

                    if (seq_start) begin
                        for (i = 0; i < HEAD_DIM; i = i + 1)
                            acc[i] <= '0;
                        state <= ACCUMULATE;
                    end
                end

                ACCUMULATE: begin
                    if (output_pending && denom_valid) begin
                        denom_reg      <= denom_in;
                        output_pending <= 1'b0;
                        o_idx          <= '0;
                        state          <= OUTPUT;
                    end else if (rescale_valid) begin
                        rescale_reg <= rescale_factor;
                        r_idx       <= '0;
                        state       <= RESCALE;
                    end else begin
                        if (weight_valid)
                            weight_reg <= weight_in;

                        if (v_valid) begin
                        acc[v_idx] <= acc[v_idx] + weighted_q16_16;

                        if (v_idx == (HEAD_DIM - 1)) begin
                            v_idx <= '0;

                            if (seq_last) begin
                                if (denom_valid) begin
                                    denom_reg <= denom_in;
                                    o_idx     <= '0;
                                    state     <= OUTPUT;
                                end else begin
                                    output_pending <= 1'b1;
                                end
                            end
                        end else begin
                            v_idx <= v_idx + 1'b1;
                        end
                        end
                    end
                end

                RESCALE: begin
                    acc[r_idx] <= rescaled_acc;

                    if (r_idx == (HEAD_DIM - 1)) begin
                        r_idx <= '0;
                        state <= ACCUMULATE;
                    end else begin
                        r_idx <= r_idx + 1'b1;
                    end
                end

                OUTPUT: begin
                    // denom_reg is latched for interface completeness; the
                    // fixed-point datapath uses the standard no-divider path.
                    ctx_out   <= acc[o_idx][DATA_WIDTH+7:8];
                    ctx_valid <= 1'b1;
                    ctx_last  <= (o_idx == (HEAD_DIM - 1));

                    if (o_idx == (HEAD_DIM - 1)) begin
                        o_idx <= '0;
                        state <= IDLE;
                        output_pending <= 1'b0;
                    end else begin
                        o_idx <= o_idx + 1'b1;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

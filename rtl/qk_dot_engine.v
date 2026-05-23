`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Module : qk_dot_engine
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: 8-PE folded systolic dot product, Q8.8 fixed-point, HEAD_DIM=64
// -----------------------------------------------------------------------------
//
// Architecture overview
// ---------------------
// HEAD_DIM=64 elements are processed in FOLD_COUNT=8 folds of PE_COUNT=8
// parallel PEs.  Within fold f, PE j handles element index f*8+j:
//
//   cycle  | fold_cnt | elem_cnt | active PE
//   -------|----------|----------|----------
//     1    |    0     |    0     |  PE 0  -> Q[0]  * K[0]
//     2    |    0     |    1     |  PE 1  -> Q[1]  * K[1]
//    ...   |   ...    |   ...    |  ...
//     8    |    0     |    7     |  PE 7  -> Q[7]  * K[7]
//     9    |    1     |    0     |  PE 0  -> Q[8]  * K[8]
//    ...
//    64    |    7     |    7     |  PE 7  -> Q[63] * K[63]
//    65    |    -     |    -     |  dot_valid, dot_result valid
//
// Each PE accumulates one product per fold into a 32-bit (Q16.16) register.
// After all 64 k_valid cycles the 8 pe_acc values are reduced through a
// combinatorial 3-level adder tree and registered one cycle later.
// -----------------------------------------------------------------------------
module qk_dot_engine #(
    parameter HEAD_DIM   = 64,
    parameter DATA_WIDTH = 16,
    parameter PE_COUNT   = 8,
    parameter ACC_WIDTH  = 32,
    parameter MAX_BATCH  = 8
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Q loading (batch-aware)
    input  logic [DATA_WIDTH-1:0]         q_data,
    input  logic                          q_valid,
    input  logic [$clog2(HEAD_DIM)-1:0]   q_addr,
    input  logic [$clog2(MAX_BATCH)-1:0]  q_batch_id,

    // K streaming
    input  logic [DATA_WIDTH-1:0]         k_data,
    input  logic                          k_valid,
    input  logic                          k_start,  // one-cycle pulse, resets and arms engine
    input  logic [$clog2(MAX_BATCH)-1:0]  k_batch_id,

    // Output
    output logic [ACC_WIDTH-1:0]          dot_result,
    output logic                          dot_valid   // one-cycle pulse, cycle 65 after k_start
);

    // -------------------------------------------------------------------------
    // Derived constants
    // -------------------------------------------------------------------------
    localparam FOLD_COUNT = HEAD_DIM / PE_COUNT;     // 8 folds
    localparam FOLD_BITS  = $clog2(FOLD_COUNT);      // 3 bits to hold 0..7
    localparam ELEM_BITS  = $clog2(PE_COUNT);        // 3 bits to hold 0..7
    localparam MUL_WIDTH  = 2 * DATA_WIDTH;          // 32-bit Q16.16 product

    // -------------------------------------------------------------------------
    // Q register files: MAX_BATCH independent banks, HEAD_DIM entries each
    // -------------------------------------------------------------------------
    // q_rf[batch][addr] -- written any time, independent of K streaming
    logic [DATA_WIDTH-1:0] q_rf [MAX_BATCH-1:0][HEAD_DIM-1:0];

    always_ff @(posedge clk) begin
        if (q_valid)
            q_rf[q_batch_id][q_addr] <= q_data;
    end

    // -------------------------------------------------------------------------
    // K-stream control: fold counter, element counter, active flag
    // -------------------------------------------------------------------------
    logic                  active;    // high while consuming K elements
    logic [FOLD_BITS-1:0]  fold_cnt;  // current fold (0..FOLD_COUNT-1)
    logic [ELEM_BITS-1:0]  elem_cnt;  // current element within fold (0..PE_COUNT-1)

    // fires on the last k_valid of the last fold; pe_acc is being written this cycle
    logic compute_done;
    assign compute_done = active && k_valid &&
                          (fold_cnt == (FOLD_COUNT - 1)) &&
                          (elem_cnt == (PE_COUNT  - 1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active   <= 1'b0;
            fold_cnt <= '0;
            elem_cnt <= '0;
        end else if (k_start) begin
            // k_start takes priority; engine is armed for the next k_valid
            active   <= 1'b1;
            fold_cnt <= '0;
            elem_cnt <= '0;
        end else if (active && k_valid) begin
            if (elem_cnt == (PE_COUNT - 1)) begin
                elem_cnt <= '0;
                if (fold_cnt == (FOLD_COUNT - 1)) begin
                    fold_cnt <= '0;
                    active   <= 1'b0;   // last element consumed; disarm
                end else begin
                    fold_cnt <= fold_cnt + 1'b1;
                end
            end else begin
                elem_cnt <= elem_cnt + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // PE array: PE_COUNT=8 processing elements
    //
    // PE j fires every fold when elem_cnt reaches j.  It reads Q element
    // q_rf[k_batch_id][fold*8 + j] and multiplies by the arriving k_data,
    // accumulating the Q16.16 product into pe_acc[j].
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] pe_acc [PE_COUNT-1:0]; // Q16.16 partial sums

    genvar j;
    generate
        for (j = 0; j < PE_COUNT; j++) begin : gen_pe
            // Q element for this PE: lane j of the current fold
            logic signed [DATA_WIDTH-1:0] q_elem;   // Q8.8 value
            logic signed [MUL_WIDTH-1:0]  mul_prod; // Q8.8 * Q8.8 -> Q16.16

            // {fold_cnt, j[2:0]} == fold_cnt*PE_COUNT + j, the element index
            assign q_elem   = $signed(q_rf[k_batch_id][{fold_cnt, ELEM_BITS'(j)}]);
            assign mul_prod = q_elem * $signed(k_data);

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    pe_acc[j] <= '0;
                end else if (k_start) begin
                    pe_acc[j] <= '0;
                end else if (active && k_valid && (elem_cnt == ELEM_BITS'(j))) begin
                    pe_acc[j] <= pe_acc[j] + mul_prod;
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Adder tree: 8 -> 4 -> 2 -> 1, fully combinatorial
    //
    // Levels are:
    //   L1: 4 x (2-input add) over pairs {0,1}, {2,3}, {4,5}, {6,7}
    //   L2: 2 x (2-input add) over L1 pairs
    //   L3: 1 x (2-input add) -> final 32-bit sum
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] add_l1 [3:0]; // 4 partial sums
    logic signed [ACC_WIDTH-1:0] add_l2 [1:0]; // 2 partial sums
    logic signed [ACC_WIDTH-1:0] add_l3;        // final dot product (combinatorial)

    always_comb begin
        add_l1[0] = pe_acc[0] + pe_acc[1];
        add_l1[1] = pe_acc[2] + pe_acc[3];
        add_l1[2] = pe_acc[4] + pe_acc[5];
        add_l1[3] = pe_acc[6] + pe_acc[7];
        add_l2[0] = add_l1[0] + add_l1[1];
        add_l2[1] = add_l1[2] + add_l1[3];
        add_l3    = add_l2[0] + add_l2[1];
    end

    // -------------------------------------------------------------------------
    // Output register: adder tree is sampled one cycle after compute_done
    //
    // Timing:
    //   cycle 64: last k_valid consumed; pe_acc[7] written this edge;
    //             compute_done=1; done_d1 captures compute_done.
    //   cycle 65: done_d1=1; all pe_acc are now final;
    //             add_l3 (comb) is correct; dot_result and dot_valid registered.
    // -------------------------------------------------------------------------
    logic done_d1; // compute_done delayed one cycle so all pe_acc are settled

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_d1    <= 1'b0;
            dot_valid  <= 1'b0;
            dot_result <= '0;
        end else begin
            done_d1   <= compute_done;
            dot_valid <= done_d1;           // one-cycle pulse on cycle 65
            if (done_d1)
                dot_result <= add_l3;       // capture adder tree over final pe_acc
        end
    end

endmodule

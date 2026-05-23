// -----------------------------------------------------------------------------
// Module : tb_softmax_engine
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: xsim testbench for softmax_engine.v
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_softmax_engine;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam SCORE_WIDTH = 16;
    localparam ACC_WIDTH   = 32;
    localparam EXP_WIDTH   = 16;

    localparam logic [EXP_WIDTH-1:0] EXP_ONE = 16'h7FFF;
    localparam logic [ACC_WIDTH-1:0] DEN_ONE = 32'h0000_7FFF;

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic                     clk;
    logic                     rst_n;
    logic [SCORE_WIDTH-1:0]   score_in;
    logic                     score_valid;
    logic                     seq_last;
    logic                     seq_start;
    logic [EXP_WIDTH-1:0]     weight_out;
    logic                     weight_valid;
    logic                     rescale_valid;
    logic [EXP_WIDTH-1:0]     rescale_factor;
    logic [ACC_WIDTH-1:0]     denom_out;
    logic                     denom_valid;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    softmax_engine #(
        .SCORE_WIDTH(SCORE_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .EXP_WIDTH  (EXP_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .score_in      (score_in),
        .score_valid   (score_valid),
        .seq_last      (seq_last),
        .seq_start     (seq_start),
        .weight_out    (weight_out),
        .weight_valid  (weight_valid),
        .rescale_valid (rescale_valid),
        .rescale_factor(rescale_factor),
        .denom_out     (denom_out),
        .denom_valid   (denom_valid)
    );

    // -------------------------------------------------------------------------
    // Clock: 10ns period (100 MHz)
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // PASS/FAIL counters and captured output state
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;
    integer rescale_count;
    bit     denom_seen;

    logic [EXP_WIDTH-1:0] last_weight;
    logic [EXP_WIDTH-1:0] last_rescale_factor;
    logic [ACC_WIDTH-1:0] last_denom;

    // -------------------------------------------------------------------------
    // Task: pass/fail helper
    // -------------------------------------------------------------------------
    task automatic check_cond(
        input bit    cond,
        input string label
    );
        if (cond) begin
            $display("PASS [%s]", label);
            pass_cnt++;
        end else begin
            $display("FAIL [%s]", label);
            fail_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: send_score
    //   Drives one score_valid cycle.  Outputs are expected two cycles later.
    // -------------------------------------------------------------------------
    task automatic send_score(
        input logic [SCORE_WIDTH-1:0] score_val,
        input logic                   is_last
    );
        @(negedge clk);
        score_in    = score_val;
        score_valid = 1'b1;
        seq_last    = is_last;
        @(negedge clk);
        score_valid = 1'b0;
        seq_last    = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Task: pulse_seq_start
    // -------------------------------------------------------------------------
    task automatic pulse_seq_start;
        @(negedge clk);
        seq_start = 1'b1;
        denom_seen = 1'b0;
        @(negedge clk);
        seq_start = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Task: wait_for_weight
    // -------------------------------------------------------------------------
    task automatic wait_for_weight(
        output logic [EXP_WIDTH-1:0] weight_val,
        input  string                label
    );
        integer i;
        bit found;

        found = 1'b0;
        for (i = 0; i < 8 && !found; i++) begin
            @(posedge clk);
            #1;
            if (rescale_valid) begin
                rescale_count       = rescale_count + 1;
                last_rescale_factor = rescale_factor;
            end
            if (weight_valid) begin
                weight_val = weight_out;
                last_weight = weight_out;
                found = 1'b1;
            end
            if (denom_valid) begin
                last_denom = denom_out;
                denom_seen = 1'b1;
            end
        end

        if (!found) begin
            weight_val = '0;
            $display("FAIL [%s]: weight_valid timeout", label);
            fail_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: wait_for_denom
    // -------------------------------------------------------------------------
    task automatic wait_for_denom(
        output logic [ACC_WIDTH-1:0] denom_val,
        input  string                label
    );
        integer i;
        bit found;

        found = 1'b0;
        if (denom_seen) begin
            denom_val = last_denom;
            denom_seen = 1'b0;
            found = 1'b1;
        end

        for (i = 0; i < 8 && !found; i++) begin
            @(posedge clk);
            #1;
            if (denom_valid) begin
                denom_val = denom_out;
                last_denom = denom_out;
                denom_seen = 1'b0;
                found = 1'b1;
            end
        end

        if (!found) begin
            denom_val = '0;
            $display("FAIL [%s]: denom_valid timeout", label);
            fail_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: reset rescale observation window
    // -------------------------------------------------------------------------
    task automatic clear_rescale_window;
        @(negedge clk);
        rescale_count       = 0;
        last_rescale_factor = '0;
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    initial begin
        logic [EXP_WIDTH-1:0] w0;
        logic [EXP_WIDTH-1:0] w1;
        logic [EXP_WIDTH-1:0] wlast;
        logic [ACC_WIDTH-1:0] den0;
        logic [ACC_WIDTH-1:0] den1;
        logic [ACC_WIDTH-1:0] den2;
        logic [SCORE_WIDTH-1:0] dec_scores [0:7];
        integer i;

        rst_n               = 1'b0;
        score_in            = '0;
        score_valid         = 1'b0;
        seq_last            = 1'b0;
        seq_start           = 1'b0;
        pass_cnt            = 0;
        fail_cnt            = 0;
        rescale_count       = 0;
        denom_seen          = 1'b0;
        last_weight         = '0;
        last_rescale_factor = '0;
        last_denom          = '0;

        dec_scores[0] = 16'h0000;
        dec_scores[1] = 16'hF000;
        dec_scores[2] = 16'hE000;
        dec_scores[3] = 16'hD000;
        dec_scores[4] = 16'hC000;
        dec_scores[5] = 16'hB000;
        dec_scores[6] = 16'hA000;
        dec_scores[7] = 16'h9000;

        repeat(4) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 1: Single score, trivial softmax
        // -----------------------------------------------------------------
        $display("TEST 1: single score");
        pulse_seq_start();
        clear_rescale_window();
        send_score(16'h0000, 1'b1);
        wait_for_weight(w0, "T1:weight_valid");
        wait_for_denom(den0, "T1:denom_valid");
        check_cond(w0 === EXP_ONE, "T1:weight_out == 0x7FFF");
        check_cond(den0 === DEN_ONE, "T1:denom_out == 0x00007FFF");
        check_cond(rescale_count == 0, "T1:no rescale_valid");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 2: Two scores, no rescale on second score
        // -----------------------------------------------------------------
        $display("TEST 2: two scores, no rescale");
        pulse_seq_start();
        clear_rescale_window();
        send_score(16'h0000, 1'b0);
        wait_for_weight(w0, "T2:weight0_valid");
        clear_rescale_window();
        send_score(16'hE000, 1'b1);
        wait_for_weight(w1, "T2:weight1_valid");
        wait_for_denom(den1, "T2:denom_valid");
        check_cond(w0 === EXP_ONE, "T2:weight0 == 0x7FFF");
        check_cond((w1 != 16'h0000) && (w1 < EXP_ONE), "T2:weight1 nonzero and below 1.0");
        check_cond(den1 > {{(ACC_WIDTH-EXP_WIDTH){1'b0}}, w1}, "T2:denom greater than weight1");
        check_cond(rescale_count == 0, "T2:no rescale_valid on score1");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 3: Rescale fires when new max found
        // -----------------------------------------------------------------
        $display("TEST 3: rescale on new max");
        pulse_seq_start();
        clear_rescale_window();
        send_score(16'hE000, 1'b0);
        wait_for_weight(w0, "T3:weight0_valid");
        clear_rescale_window();
        send_score(16'h0000, 1'b1);
        wait_for_weight(w1, "T3:weight1_valid");
        wait_for_denom(den1, "T3:denom_valid");
        check_cond(w1 === EXP_ONE, "T3:weight1 == 0x7FFF");
        check_cond(rescale_count == 1, "T3:rescale_valid asserted once");
        check_cond((last_rescale_factor != 16'h0000) && (last_rescale_factor < EXP_ONE),
                   "T3:rescale_factor nonzero and below 1.0");
        check_cond(den1 > 32'h0000_0000, "T3:denom_out greater than zero");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 4: Monotonically decreasing sequence, no rescale ever
        // -----------------------------------------------------------------
        $display("TEST 4: monotonically decreasing sequence");
        pulse_seq_start();
        clear_rescale_window();
        for (i = 0; i < 8; i++) begin
            send_score(dec_scores[i], (i == 7));
            wait_for_weight(wlast, "T4:weight_valid");
        end
        wait_for_denom(den1, "T4:denom_valid");
        check_cond(rescale_count == 0, "T4:no rescale_valid");
        check_cond(den1 > {{(ACC_WIDTH-EXP_WIDTH){1'b0}}, wlast},
                   "T4:denom greater than last weight");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 5: seq_start resets state
        // -----------------------------------------------------------------
        $display("TEST 5: seq_start resets state");
        pulse_seq_start();
        clear_rescale_window();
        send_score(16'h0000, 1'b1);
        wait_for_weight(w0, "T5:first_weight_valid");
        wait_for_denom(den1, "T5:first_denom_valid");

        pulse_seq_start();
        clear_rescale_window();
        send_score(16'h0000, 1'b1);
        wait_for_weight(w1, "T5:second_weight_valid");
        wait_for_denom(den2, "T5:second_denom_valid");

        check_cond(den1 === den2, "T5:denoms identical after seq_start");
        check_cond(w0 === w1, "T5:weights identical after seq_start");
        check_cond(den1 === DEN_ONE, "T5:reset sequence denom == 0x00007FFF");

        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        $display("--------------------------------------------------");
        $display("Results: %0d PASS, %0d FAIL", pass_cnt, fail_cnt);
        $display("--------------------------------------------------");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog: abort if simulation hangs
    // -------------------------------------------------------------------------
    initial begin
        #100000;
        $display("FAIL [watchdog]: simulation exceeded time limit");
        $finish;
    end

endmodule

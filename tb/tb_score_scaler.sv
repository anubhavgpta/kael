// -----------------------------------------------------------------------------
// Module : tb_score_scaler
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: xsim testbench for score_scaler.v
// -----------------------------------------------------------------------------
//
// Tests
// -----
// T1: 30.0   in Q16.16 (0x001E0000) -> positive overflow -> 0x7FFF
// T2:  0.5   in Q16.16 (0x00008000) -> 0.5/8 = 0.0625  -> 0x0800
// T3: -0.5   in Q16.16 (0xFFFF8000) -> -0.5/8 = -0.0625 -> 0xF800
// T4: 64.0   in Q16.16 (0x00400000) -> positive overflow -> 0x7FFF
// T5: large negative    (0xC0000000) -> negative overflow -> 0x8000
// T6: timing - score_valid asserts exactly 1 cycle after dot_valid
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_score_scaler;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam IN_WIDTH  = 32;
    localparam OUT_WIDTH = 16;

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic                  clk;
    logic                  rst_n;
    logic [IN_WIDTH-1:0]   dot_in;
    logic                  dot_valid;
    logic [OUT_WIDTH-1:0]  score_out;
    logic                  score_valid;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    score_scaler #(
        .IN_WIDTH (IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .dot_in     (dot_in),
        .dot_valid  (dot_valid),
        .score_out  (score_out),
        .score_valid(score_valid)
    );

    // -------------------------------------------------------------------------
    // Clock: 10ns period (100 MHz)
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Pass/fail counters
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;

    // -------------------------------------------------------------------------
    // Task: check_result
    //   Drives dot_in and dot_valid on the negedge, waits for the capture
    //   posedge, then checks score_out and score_valid after a 1ns propagation
    //   delay (past the NBA region).  Deasserts dot_valid on the next negedge.
    //
    //   Latency model: always_ff registers outputs on the posedge that captures
    //   dot_valid.  At posedge+1ns the NBA has settled and outputs are stable.
    // -------------------------------------------------------------------------
    task automatic check_result(
        input logic [IN_WIDTH-1:0]  din,
        input logic [OUT_WIDTH-1:0] expected,
        input string                label
    );
        @(negedge clk);
        dot_in    = din;
        dot_valid = 1'b1;

        @(posedge clk); // DUT captures inputs at this edge
        #1;             // allow NBA region to complete

        if (!score_valid) begin
            $display("FAIL [%s]: score_valid not asserted", label);
            fail_cnt++;
        end else if (score_out !== expected) begin
            $display("FAIL [%s]: expected 0x%04X  got 0x%04X", label, expected, score_out);
            fail_cnt++;
        end else begin
            $display("PASS [%s]: score_out=0x%04X", label, score_out);
            pass_cnt++;
        end

        @(negedge clk);
        dot_valid = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    initial begin
        rst_n     = 1'b0;
        dot_in    = '0;
        dot_valid = 1'b0;
        pass_cnt  = 0;
        fail_cnt  = 0;

        repeat(4) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T1: Positive overflow
        //   30.0 in Q16.16 = 0x001E0000
        //   30.0 / 8 = 3.75, exceeds Q1.15 max -- clamp to 0x7FFF
        // -----------------------------------------------------------------
        check_result(32'h001E0000, 16'h7FFF, "T1:sat_pos_30.0");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T2: In-range positive
        //   0.5 in Q16.16 = 0x00008000  (0.5 * 65536 = 32768)
        //   0.5 / 8 = 0.0625
        //   Q1.15: 0.0625 * 32768 = 2048 = 0x0800
        // -----------------------------------------------------------------
        check_result(32'h00008000, 16'h0800, "T2:0.5_to_0.0625");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T3: In-range negative
        //   -0.5 in Q16.16 = 0xFFFF8000  (-0.5 * 65536 = -32768)
        //   -0.5 / 8 = -0.0625
        //   Q1.15: -0.0625 * 32768 = -2048 = 0xF800
        // -----------------------------------------------------------------
        check_result(32'hFFFF8000, 16'hF800, "T3:-0.5_to_-0.0625");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T4: Strong positive overflow
        //   64.0 in Q16.16 = 0x00400000  (64 * 65536 = 4194304)
        //   64.0 / 8 = 8.0, far above Q1.15 max -- clamp to 0x7FFF
        // -----------------------------------------------------------------
        check_result(32'h00400000, 16'h7FFF, "T4:sat_pos_64.0");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T5: Negative overflow
        //   0xC0000000 as signed int32 = -1073741824
        //   In Q16.16 that is -16384.0 (not -64.0; the spec comment was off)
        //   Either way, far below Q1.15 min -- clamp to 0x8000
        // -----------------------------------------------------------------
        check_result(32'hC0000000, 16'h8000, "T5:sat_neg_large");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // T6: Timing
        //   score_valid must appear exactly 1 cycle after dot_valid.
        //   Use a cycle counter: drive dot_valid, count posedges with #1
        //   NBA guard until score_valid is seen; verify count == 1.
        // -----------------------------------------------------------------
        begin : test6
            integer cycles;
            logic   found;

            cycles = 0;
            found  = 1'b0;

            @(negedge clk);
            dot_in    = 32'h00008000; // 0.5 in Q16.16, non-trivial but in-range
            dot_valid = 1'b1;

            // Scan up to 4 posedges; score_valid must appear at cycle 1
            begin : t6_scan
                integer s;
                for (s = 0; s < 4 && !found; s++) begin
                    @(posedge clk);
                    #1;             // past NBA so registered outputs are visible
                    cycles++;
                    if (score_valid)
                        found = 1'b1;
                end
            end

            @(negedge clk);
            dot_valid = 1'b0;

            if (!found) begin
                $display("FAIL [T6:timing]: score_valid never asserted within 4 cycles");
                fail_cnt++;
            end else if (cycles !== 1) begin
                $display("FAIL [T6:timing]: expected score_valid at cycle 1, got cycle %0d",
                         cycles);
                fail_cnt++;
            end else begin
                $display("PASS [T6:timing]: score_valid at cycle %0d (1-cycle latency correct)",
                         cycles);
                pass_cnt++;
            end
        end

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
        #50000;
        $display("FAIL [watchdog]: simulation exceeded time limit");
        $finish;
    end

endmodule
